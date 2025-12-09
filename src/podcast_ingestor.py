"""
Advanced Podcast Ingestor for viral content creation.
Tracks episodes, analyzes viral potential, manages downloads.
"""

import sqlite3
import logging
import hashlib
import requests
import feedparser
import re
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
    guest_names: Optional[List[str]] = None

    # Media/visual content
    thumbnail_url: Optional[str] = None
    thumbnail_local_path: Optional[str] = None

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
        else:
            data['youtube_urls'] = None
        if self.guest_names:
            data['guest_names'] = ','.join(self.guest_names)
        else:
            data['guest_names'] = None
        # Ensure thumbnail fields exist
        if 'thumbnail_url' not in data:
            data['thumbnail_url'] = None
        if 'thumbnail_local_path' not in data:
            data['thumbnail_local_path'] = None
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
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_podcast_name ON episodes (podcast_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON episodes (processing_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON episodes (published_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viral_score ON episodes (viral_score)")
            
            # Migration: Add error_message column if it doesn't exist
            try:
                conn.execute("SELECT error_message FROM episodes LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("Adding error_message column to episodes table")
                conn.execute("ALTER TABLE episodes ADD COLUMN error_message TEXT")

            # Migration: Add thumbnail fields if they don't exist
            try:
                conn.execute("SELECT thumbnail_url FROM episodes LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("Adding thumbnail_url column to episodes table")
                conn.execute("ALTER TABLE episodes ADD COLUMN thumbnail_url TEXT")

            try:
                conn.execute("SELECT thumbnail_local_path FROM episodes LIMIT 1")
            except sqlite3.OperationalError:
                logger.info("Adding thumbnail_local_path column to episodes table")
                conn.execute("ALTER TABLE episodes ADD COLUMN thumbnail_local_path TEXT")
    
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
                    data['engagement_metrics'] = json.dumps(data['engagement_metrics'], ensure_ascii=False)
                
                # Insert or update - all 23 columns explicitly listed
                conn.execute("""
                    INSERT OR REPLACE INTO episodes
                    (episode_id, title, podcast_name, audio_url, published_date, description,
                     episode_number, duration, file_size, youtube_urls, viral_score, processing_status,
                     transcription_method, content_extracted, tweets_generated, engagement_metrics,
                     error_message, created_at, updated_at, insights_count, tweets_generated_count,
                     thumbnail_url, thumbnail_local_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            COALESCE((SELECT created_at FROM episodes WHERE episode_id = ?), CURRENT_TIMESTAMP),
                            CURRENT_TIMESTAMP,
                            ?, ?, ?, ?)
                """, (
                    data['episode_id'], data['title'], data['podcast_name'],
                    data['audio_url'], data['published_date'], data['description'],
                    data['episode_number'], data['duration'], data['file_size'],
                    data['youtube_urls'], data['viral_score'], data['processing_status'],
                    data['transcription_method'], data['content_extracted'],
                    data['tweets_generated'], data['engagement_metrics'],
                    data.get('error_message'),
                    data['episode_id'],  # For the COALESCE query (created_at)
                    0,  # insights_count
                    0,  # tweets_generated_count
                    data.get('thumbnail_url'),
                    data.get('thumbnail_local_path')
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
                db_fields = ['created_at', 'updated_at', 'error_message', 'insights_count', 'tweets_generated_count']
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

    def get_episodes_without_thumbnails(self, limit: int = 20) -> List[PodcastEpisode]:
        """
        Get episodes that don't have thumbnails yet.

        Args:
            limit: Maximum number of episodes to return

        Returns:
            List of PodcastEpisode objects needing thumbnails
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM episodes
                WHERE thumbnail_local_path IS NULL
                ORDER BY published_date DESC
                LIMIT ?
            """, (limit,))

            episodes = []
            for row in cursor.fetchall():
                episode_data = dict(row)

                # Remove database-specific fields
                db_fields = ['created_at', 'updated_at', 'error_message', 'insights_count', 'tweets_generated_count']
                for field in db_fields:
                    episode_data.pop(field, None)

                # Convert ISO string back to datetime
                episode_data['published_date'] = datetime.fromisoformat(episode_data['published_date'])

                # Convert YouTube URLs string back to list
                if episode_data.get('youtube_urls'):
                    episode_data['youtube_urls'] = episode_data['youtube_urls'].split(',')
                else:
                    episode_data['youtube_urls'] = None

                # Convert engagement metrics back to dict
                if episode_data.get('engagement_metrics'):
                    import json
                    episode_data['engagement_metrics'] = json.loads(episode_data['engagement_metrics'])
                else:
                    episode_data['engagement_metrics'] = None

                episodes.append(PodcastEpisode(**episode_data))

            return episodes

    def update_thumbnail_path(self, episode_id: str, thumbnail_path: str):
        """
        Update the thumbnail local path for an episode.

        Args:
            episode_id: Episode ID
            thumbnail_path: Local path to the cached thumbnail
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE episodes
                   SET thumbnail_local_path = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE episode_id = ?""",
                (thumbnail_path, episode_id)
            )

    def backfill_episode_numbers(self, feed_urls: List[str] = None) -> int:
        """
        Backfill episode numbers for existing episodes that have NULL episode_number.
        First tries to extract from titles using regex patterns.
        If feed_urls provided, also tries to match with RSS itunes_episode metadata.

        Args:
            feed_urls: Optional list of RSS feed URLs to fetch itunes_episode from

        Returns:
            Number of episodes updated
        """
        import re

        patterns = [
            r'Episode\s+(\d+)',
            r'Ep\.?\s+(\d+)',
            r'#(\d+)',
            r'^(\d+)\s*[-–:.]',  # Numbers at start like "123 - Title" or "123: Title"
        ]

        updated_count = 0
        episodes_to_update = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT episode_id, title FROM episodes
                   WHERE episode_number IS NULL OR episode_number = ''"""
            )

            missing_episodes = list(cursor)

            # Step 1: Try to extract from titles
            title_matches = {}
            for row in missing_episodes:
                episode_id = row['episode_id']
                title = row['title']

                for pattern in patterns:
                    match = re.search(pattern, title, re.IGNORECASE)
                    if match:
                        title_matches[episode_id] = match.group(1)
                        break

            # Step 2: Try to fetch from RSS feeds if provided
            rss_matches = {}
            if feed_urls:
                import feedparser
                for feed_url in feed_urls:
                    try:
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries:
                            title = entry.get('title', '').strip()
                            itunes_ep = getattr(entry, 'itunes_episode', None)

                            if itunes_ep:
                                # Try to match by title
                                for row in missing_episodes:
                                    db_title = row['title']
                                    if db_title == title or title in db_title or db_title in title:
                                        rss_matches[row['episode_id']] = str(itunes_ep)
                                        break
                    except Exception as e:
                        logger.warning(f"Error fetching RSS feed {feed_url}: {e}")

            # Combine matches (title takes precedence)
            for row in missing_episodes:
                episode_id = row['episode_id']
                episode_number = title_matches.get(episode_id) or rss_matches.get(episode_id)
                if episode_number:
                    episodes_to_update.append((episode_number, episode_id))

            # Batch update
            if episodes_to_update:
                conn.executemany(
                    """UPDATE episodes
                       SET episode_number = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE episode_id = ?""",
                    episodes_to_update
                )
                updated_count = len(episodes_to_update)
                logger.info(f"Backfilled episode numbers for {updated_count} episodes")

        return updated_count


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
                episode = self._parse_entry(entry, podcast_name, feed)
                if episode:
                    episodes.append(episode)
            
            return episodes
            
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
            return []
    
    def _parse_entry(self, entry, podcast_name: str, feed=None) -> Optional[PodcastEpisode]:
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

            # Extract guest names from title
            title = entry.get('title', 'Unknown Title').strip()
            guest_names = self._extract_guest_names(title)

            # Extract thumbnail URL
            thumbnail_url = self._extract_thumbnail_url(entry, feed)

            episode = PodcastEpisode(
                title=title,
                podcast_name=podcast_name.strip(),
                audio_url=audio_url,
                published_date=published_date,
                description=entry.get('description', '').strip(),
                episode_number=self._extract_episode_number(entry),
                duration=duration,
                youtube_urls=youtube_urls if youtube_urls else None,
                guest_names=guest_names if guest_names else None,
                thumbnail_url=thumbnail_url
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

    def _extract_thumbnail_url(self, entry, feed=None) -> Optional[str]:
        """
        Extract thumbnail URL from RSS entry or feed.

        Priority order:
        1. Episode-specific image (itunes:image on entry)
        2. media:content or media:thumbnail
        3. Podcast channel image (fallback)

        Args:
            entry: RSS entry object
            feed: Full feed object (for channel image fallback)

        Returns:
            Thumbnail URL or None
        """
        # 1. Episode-specific itunes:image
        if hasattr(entry, 'image') and entry.image:
            if isinstance(entry.image, dict):
                href = entry.image.get('href')
                if href:
                    return href
            elif isinstance(entry.image, str):
                return entry.image

        # 2. itunes_image attribute
        if hasattr(entry, 'itunes_image'):
            img = entry.itunes_image
            if isinstance(img, dict):
                href = img.get('href')
                if href:
                    return href
            elif isinstance(img, str):
                return img

        # 3. media:content (look for image type)
        if hasattr(entry, 'media_content'):
            for media in entry.media_content:
                medium = media.get('medium', '')
                mime_type = media.get('type', '')
                if medium == 'image' or mime_type.startswith('image'):
                    url = media.get('url')
                    if url:
                        return url

        # 4. media:thumbnail
        if hasattr(entry, 'media_thumbnail'):
            for thumb in entry.media_thumbnail:
                url = thumb.get('url')
                if url:
                    return url

        # 5. Fallback: podcast channel image
        if feed and hasattr(feed, 'feed'):
            channel = feed.feed

            # Channel image
            if hasattr(channel, 'image') and channel.image:
                if isinstance(channel.image, dict):
                    href = channel.image.get('href')
                    if href:
                        return href

            # Channel itunes:image
            if hasattr(channel, 'itunes_image'):
                img = channel.itunes_image
                if isinstance(img, dict):
                    href = img.get('href')
                    if href:
                        return href
                elif isinstance(img, str):
                    return img

        return None

    def _extract_guest_names(self, title: str) -> List[str]:
        """
        Extract guest names from episode title.

        Common patterns:
        - "Episode Title | Guest Name"
        - "Episode Title with Guest Name"
        - "Guest Name: Episode Title"
        - "Episode Title ft. Guest Name"
        - "#NNN – Guest Name" (Lex Fridman style)

        Args:
            title: Episode title

        Returns:
            List of extracted guest names
        """
        guests = []

        # Pattern 1: "Title | Guest Name"
        if '|' in title:
            parts = title.split('|')
            if len(parts) >= 2:
                potential_guest = parts[1].strip()
                # Remove common suffixes
                potential_guest = re.sub(r'\s+(PhD|MD|Dr\.|M\.D\.|Ph\.D\.).*$', '', potential_guest)
                if potential_guest and len(potential_guest) > 3:
                    guests.append(potential_guest)

        # Pattern 2: "with Guest Name"
        with_match = re.search(r'\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', title)
        if with_match:
            guests.append(with_match.group(1).strip())

        # Pattern 3: "ft. Guest Name" or "feat. Guest Name"
        feat_match = re.search(r'\b(?:ft\.|feat\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', title)
        if feat_match:
            guests.append(feat_match.group(1).strip())

        # Pattern 4: "Guest Name:" at start
        colon_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+):', title)
        if colon_match:
            guests.append(colon_match.group(1).strip())

        # Pattern 5: "#NNN – Guest Name" (Lex Fridman style)
        lex_match = re.match(r'^#\d+\s*[–-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', title)
        if lex_match:
            potential_guest = lex_match.group(1).strip()
            # Remove topic suffix if present (e.g., "Name: Topic")
            if ':' in potential_guest:
                potential_guest = potential_guest.split(':')[0].strip()
            guests.append(potential_guest)

        # Deduplicate and filter
        guests = list(set(guests))
        guests = [g for g in guests if self._is_likely_person_name(g)]

        return guests

    def _is_likely_person_name(self, name: str) -> bool:
        """
        Check if a string is likely a person's name.

        Args:
            name: Potential name string

        Returns:
            True if likely a person name
        """
        # Must have at least first and last name
        parts = name.split()
        if len(parts) < 2:
            return False

        # Each part should start with capital letter
        if not all(part[0].isupper() for part in parts if part):
            return False

        # Shouldn't be too long (likely a phrase, not a name)
        if len(parts) > 4:
            return False

        # Exclude common non-name patterns
        exclude_patterns = [
            'Episode', 'Podcast', 'Show', 'Interview', 'Talk',
            'Part', 'Series', 'Season', 'Special', 'Live'
        ]
        if any(pattern in name for pattern in exclude_patterns):
            return False

        return True

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