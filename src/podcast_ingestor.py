"""
Advanced Podcast Ingestor for viral content creation.
Tracks episodes, analyzes viral potential, manages downloads.
"""

import sqlite3
import logging
import hashlib
import requests
import feedparser
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class PodcastEpisode:
    """Enhanced episode representation with viral tracking."""
    
    # Core episode data
    title: str
    podcast_name: str
    audio_url: str
    published_date: datetime
    description: str
    
    # Enhanced metadata
    episode_number: Optional[str] = None
    duration: Optional[int] = None
    file_size: Optional[int] = None
    youtube_urls: Optional[List[str]] = None
    
    # Viral tracking
    episode_id: Optional[str] = None
    viral_score: Optional[float] = None
    processing_status: str = "pending"
    transcription_method: Optional[str] = None
    content_extracted: bool = False
    tweets_generated: int = 0
    engagement_metrics: Optional[Dict] = None
    
    def __post_init__(self):
        """Generate unique episode ID after initialization."""
        if not self.episode_id:
            # Create unique ID from podcast + title + publish date
            content = f"{self.podcast_name}:{self.title}:{self.published_date.isoformat()}"
            self.episode_id = hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        # Convert datetime to ISO string
        data['published_date'] = self.published_date.isoformat()
        # Convert lists to JSON strings
        if self.youtube_urls:
            data['youtube_urls'] = ','.join(self.youtube_urls)
        return data


class EpisodeDatabase:
    """SQLite database for episode tracking and analytics."""
    
    def __init__(self, db_path: str = "data/episodes.db"):
        """Initialize episode database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    podcast_name TEXT NOT NULL,
                    audio_url TEXT NOT NULL,
                    published_date TEXT NOT NULL,
                    description TEXT,
                    episode_number TEXT,
                    duration INTEGER,
                    file_size INTEGER,
                    youtube_urls TEXT,
                    viral_score REAL,
                    processing_status TEXT DEFAULT 'pending',
                    transcription_method TEXT,
                    content_extracted BOOLEAN DEFAULT 0,
                    tweets_generated INTEGER DEFAULT 0,
                    engagement_metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_podcast_name ON episodes (podcast_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON episodes (processing_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON episodes (published_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viral_score ON episodes (viral_score)")
    
    def episode_exists(self, episode_id: str) -> bool:
        """Check if episode already exists in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM episodes WHERE episode_id = ?", (episode_id,))
            return cursor.fetchone() is not None
    
    def save_episode(self, episode: PodcastEpisode) -> bool:
        """Save episode to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = episode.to_dict()
                
                # Convert engagement_metrics dict to JSON string if present
                if data.get('engagement_metrics'):
                    import json
                    data['engagement_metrics'] = json.dumps(data['engagement_metrics'])
                
                # Insert or update
                conn.execute("""
                    INSERT OR REPLACE INTO episodes 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT created_at FROM episodes WHERE episode_id = ?), CURRENT_TIMESTAMP),
                            CURRENT_TIMESTAMP)
                """, (
                    data['episode_id'], data['title'], data['podcast_name'],
                    data['audio_url'], data['published_date'], data['description'],
                    data['episode_number'], data['duration'], data['file_size'],
                    data['youtube_urls'], data['viral_score'], data['processing_status'],
                    data['transcription_method'], data['content_extracted'],
                    data['tweets_generated'], data['engagement_metrics'],
                    data['episode_id']  # For the COALESCE query
                ))
                
                logger.debug(f"Saved episode: {episode.episode_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving episode {episode.episode_id}: {e}")
            return False
    
    def get_pending_episodes(self, limit: int = 10) -> List[PodcastEpisode]:
        """Get episodes pending processing, ordered by viral potential."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM episodes 
                WHERE processing_status = 'pending'
                ORDER BY viral_score DESC, published_date DESC
                LIMIT ?
            """, (limit,))
            
            episodes = []
            for row in cursor.fetchall():
                episode_data = dict(row)
                
                # Remove database-specific fields that aren't part of PodcastEpisode
                db_fields = ['created_at', 'updated_at']
                for field in db_fields:
                    episode_data.pop(field, None)
                
                # Convert ISO string back to datetime
                episode_data['published_date'] = datetime.fromisoformat(episode_data['published_date'])
                
                # Convert YouTube URLs string back to list
                if episode_data['youtube_urls']:
                    episode_data['youtube_urls'] = episode_data['youtube_urls'].split(',')
                else:
                    episode_data['youtube_urls'] = None
                
                # Convert engagement metrics back to dict
                if episode_data['engagement_metrics']:
                    import json
                    episode_data['engagement_metrics'] = json.loads(episode_data['engagement_metrics'])
                else:
                    episode_data['engagement_metrics'] = None
                
                episodes.append(PodcastEpisode(**episode_data))
            
            return episodes
    
    def update_episode_status(self, episode_id: str, status: str, **kwargs):
        """Update episode processing status and metadata."""
        updates = {"processing_status": status}
        updates.update(kwargs)
        
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        set_clause += ", updated_at = CURRENT_TIMESTAMP"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE episodes SET {set_clause} WHERE episode_id = ?",
                (*updates.values(), episode_id)
            )


class ViralContentAnalyzer:
    """Analyzes podcast episodes for viral potential before processing."""
    
    def __init__(self):
        """Initialize viral content analyzer."""
        # Viral keywords that tend to perform well
        self.viral_keywords = {
            'controversy': ['controversial', 'shocking', 'banned', 'censored', 'secret', 'exposed'],
            'success': ['millionaire', 'billionaire', 'rich', 'successful', 'entrepreneur'],
            'productivity': ['hack', 'trick', 'secret', 'formula', 'method', 'system'],
            'psychology': ['psychology', 'mindset', 'behavior', 'manipulation', 'influence'],
            'current_events': ['2024', '2025', 'latest', 'breaking', 'new', 'recently'],
            'emotion': ['shocking', 'amazing', 'unbelievable', 'incredible', 'insane']
        }
    
    def calculate_viral_score(self, episode: PodcastEpisode) -> float:
        """Calculate viral potential score (0-1) for an episode."""
        score = 0.0
        text = f"{episode.title} {episode.description}".lower()
        
        # Keyword scoring
        keyword_score = 0
        for category, keywords in self.viral_keywords.items():
            matches = sum(1 for kw in keywords if kw in text)
            keyword_score += matches * 0.1
        
        # Title scoring (shorter, punchier titles often perform better)
        title_words = len(episode.title.split())
        if 5 <= title_words <= 12:  # Sweet spot for viral titles
            score += 0.2
        
        # Description length (moderate length descriptions work best)
        desc_words = len(episode.description.split())
        if 50 <= desc_words <= 200:
            score += 0.1
        
        # Recent episodes get priority
        days_old = (datetime.now() - episode.published_date).days
        if days_old <= 7:
            score += 0.3
        elif days_old <= 30:
            score += 0.1
        
        # YouTube presence (indicates multi-platform content)
        if episode.youtube_urls:
            score += 0.2
        
        # Cap at 1.0 and add keyword multiplier
        return min(1.0, score + keyword_score)


class PodcastIngestor:
    """Advanced podcast ingestion system with viral optimization."""
    
    def __init__(self, feed_urls: List[str], max_episodes_per_feed: int = 50):
        """
        Initialize podcast ingestor.
        
        Args:
            feed_urls: List of RSS feed URLs to monitor
            max_episodes_per_feed: Maximum episodes to fetch per feed
        """
        self.feed_urls = feed_urls
        self.max_episodes_per_feed = max_episodes_per_feed
        self.db = EpisodeDatabase()
        self.analyzer = ViralContentAnalyzer()
        
        # Create audio cache directory
        self.cache_dir = Path("cache/audio")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def discover_new_episodes(self) -> List[PodcastEpisode]:
        """Discover new episodes from all configured feeds."""
        new_episodes = []
        
        for feed_url in self.feed_urls:
            try:
                episodes = self._parse_feed(feed_url)
                new_count = 0
                
                for episode in episodes:
                    if not self.db.episode_exists(episode.episode_id):
                        # Calculate viral score before saving
                        episode.viral_score = self.analyzer.calculate_viral_score(episode)
                        
                        if self.db.save_episode(episode):
                            new_episodes.append(episode)
                            new_count += 1
                            
                        if new_count >= self.max_episodes_per_feed:
                            break
                
                logger.info(f"Found {new_count} new episodes from {feed_url}")
                
            except Exception as e:
                logger.error(f"Error parsing feed {feed_url}: {e}")
                continue
        
        # Sort by viral potential
        new_episodes.sort(key=lambda x: x.viral_score or 0, reverse=True)
        logger.info(f"Discovered {len(new_episodes)} new episodes total")
        
        return new_episodes
    
    def _parse_feed(self, feed_url: str) -> List[PodcastEpisode]:
        """Parse a single RSS feed."""
        logger.debug(f"Parsing feed: {feed_url}")
        
        try:
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                logger.warning(f"No entries found in feed: {feed_url}")
                return []
            
            podcast_name = feed.feed.get('title', 'Unknown Podcast')
            episodes = []
            
            for entry in feed.entries[:self.max_episodes_per_feed]:
                episode = self._parse_entry(entry, podcast_name)
                if episode:
                    episodes.append(episode)
            
            return episodes
            
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
            return []
    
    def _parse_entry(self, entry, podcast_name: str) -> Optional[PodcastEpisode]:
        """Parse a single RSS entry into a PodcastEpisode."""
        try:
            # Find audio URL
            audio_url = self._extract_audio_url(entry)
            if not audio_url:
                logger.debug(f"No audio URL found for: {entry.get('title', 'Unknown')}")
                return None
            
            # Parse publish date
            published_date = self._parse_publish_date(entry)
            if not published_date:
                logger.debug(f"No valid publish date for: {entry.get('title', 'Unknown')}")
                return None
            
            # Extract YouTube URLs from description
            youtube_urls = self._extract_youtube_urls(entry.get('description', ''))
            
            # Parse duration if available
            duration = self._parse_duration(entry)
            
            episode = PodcastEpisode(
                title=entry.get('title', 'Unknown Title').strip(),
                podcast_name=podcast_name.strip(),
                audio_url=audio_url,
                published_date=published_date,
                description=entry.get('description', '').strip(),
                episode_number=self._extract_episode_number(entry),
                duration=duration,
                youtube_urls=youtube_urls if youtube_urls else None
            )
            
            return episode
            
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
    
    def _extract_audio_url(self, entry) -> Optional[str]:
        """Extract audio URL from RSS entry."""
        # Check enclosures first (most common)
        if hasattr(entry, 'enclosures') and entry.enclosures:
            for enclosure in entry.enclosures:
                if enclosure.get('type', '').startswith('audio/'):
                    return enclosure.get('href')
        
        # Check links
        if hasattr(entry, 'links'):
            for link in entry.links:
                if link.get('type', '').startswith('audio/'):
                    return link.get('href')
        
        return None
    
    def _parse_publish_date(self, entry) -> Optional[datetime]:
        """Parse publish date from RSS entry."""
        date_fields = ['published_parsed', 'updated_parsed']
        
        for field in date_fields:
            if hasattr(entry, field) and getattr(entry, field):
                try:
                    time_struct = getattr(entry, field)
                    return datetime(*time_struct[:6])
                except:
                    continue
        
        return None
    
    def _extract_youtube_urls(self, text: str) -> List[str]:
        """Extract YouTube URLs from text."""
        import re
        
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'https?://youtu\.be/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'youtu\.be/([a-zA-Z0-9_-]{11})'
        ]
        
        urls = set()
        for pattern in youtube_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                urls.add(f"https://youtube.com/watch?v={match}")
        
        return list(urls)
    
    def _parse_duration(self, entry) -> Optional[int]:
        """Parse duration from RSS entry."""
        # Check iTunes duration
        if hasattr(entry, 'itunes_duration'):
            duration_str = entry.itunes_duration
            try:
                # Handle formats like "1:23:45" or "23:45" or "345"
                if ':' in duration_str:
                    parts = duration_str.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    elif len(parts) == 2:  # MM:SS
                        return int(parts[0]) * 60 + int(parts[1])
                else:
                    return int(duration_str)  # Seconds
            except:
                pass
        
        return None
    
    def _extract_episode_number(self, entry) -> Optional[str]:
        """Extract episode number from RSS entry."""
        # Check common fields
        episode_fields = ['itunes_episode', 'episode']
        
        for field in episode_fields:
            if hasattr(entry, field):
                value = getattr(entry, field)
                if value:
                    return str(value)
        
        # Try to extract from title
        import re
        title = entry.get('title', '')
        
        # Look for patterns like "Episode 123", "Ep 123", "#123"
        patterns = [
            r'Episode\s+(\d+)',
            r'Ep\.?\s+(\d+)',
            r'#(\d+)',
            r'(\d+):'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def get_episodes_for_processing(self, limit: int = 5) -> List[PodcastEpisode]:
        """Get episodes ready for processing, prioritized by viral score."""
        return self.db.get_pending_episodes(limit)
    
    def mark_episode_processing(self, episode_id: str):
        """Mark episode as currently being processed."""
        self.db.update_episode_status(episode_id, "processing")
    
    def mark_episode_completed(self, episode_id: str, **metrics):
        """Mark episode as completed with metrics."""
        self.db.update_episode_status(episode_id, "completed", **metrics)
    
    def mark_episode_failed(self, episode_id: str, error: str):
        """Mark episode as failed with error message."""
        self.db.update_episode_status(episode_id, "failed", error_message=error)