"""RSS feed parser module for extracting podcast episodes."""

import feedparser
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from .youtube_transcriber import find_youtube_urls

logger = logging.getLogger(__name__)


@dataclass
class PodcastEpisode:
    """Represents a podcast episode."""
    title: str
    podcast_name: str
    episode_number: Optional[str]
    audio_url: str
    published_date: datetime
    description: str
    duration: Optional[int] = None
    youtube_urls: Optional[List[str]] = None
    
    def __repr__(self):
        return f"Episode: {self.podcast_name} - {self.title}"
    
    def has_youtube_url(self) -> bool:
        """Check if episode has associated YouTube URLs."""
        return bool(self.youtube_urls)


class RSSFeedParser:
    """Parser for podcast RSS feeds."""
    
    def __init__(self, feed_urls: List[str], days_back: int = 7):
        """
        Initialize RSS parser.
        
        Args:
            feed_urls: List of RSS feed URLs to parse
            days_back: Number of days to look back for episodes
        """
        self.feed_urls = feed_urls
        self.days_back = days_back
        self.cutoff_date = datetime.now() - timedelta(days=days_back)
    
    def parse_feeds(self) -> List[PodcastEpisode]:
        """
        Parse all configured RSS feeds and return recent episodes.
        
        Returns:
            List of PodcastEpisode objects from recent days
        """
        all_episodes = []
        
        for feed_url in self.feed_urls:
            try:
                episodes = self._parse_single_feed(feed_url)
                all_episodes.extend(episodes)
                logger.info(f"Parsed {len(episodes)} episodes from {feed_url}")
            except Exception as e:
                logger.error(f"Error parsing feed {feed_url}: {e}")
                continue
        
        return sorted(all_episodes, key=lambda x: x.published_date, reverse=True)
    
    def _parse_single_feed(self, feed_url: str) -> List[PodcastEpisode]:
        """
        Parse a single RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            List of PodcastEpisode objects
        """
        feed = feedparser.parse(feed_url)
        
        if feed.bozo:
            raise ValueError(f"Invalid RSS feed: {feed.bozo_exception}")
        
        podcast_name = feed.feed.get('title', 'Unknown Podcast')
        episodes = []
        
        for entry in feed.entries:
            try:
                pub_date = self._parse_date(entry.get('published_parsed'))
                
                if pub_date < self.cutoff_date:
                    continue
                
                audio_url = self._extract_audio_url(entry)
                if not audio_url:
                    logger.warning(f"No audio URL found for episode: {entry.get('title')}")
                    continue
                
                episode = PodcastEpisode(
                    title=entry.get('title', 'Untitled'),
                    podcast_name=podcast_name,
                    episode_number=self._extract_episode_number(entry),
                    audio_url=audio_url,
                    published_date=pub_date,
                    description=entry.get('summary', ''),
                    duration=self._extract_duration(entry),
                    youtube_urls=self._extract_youtube_urls(entry)
                )
                
                episodes.append(episode)
                
            except Exception as e:
                logger.error(f"Error parsing episode: {e}")
                continue
        
        return episodes
    
    def _parse_date(self, time_struct) -> datetime:
        """Convert time struct to datetime."""
        if time_struct:
            return datetime(*time_struct[:6])
        return datetime.now()
    
    def _extract_audio_url(self, entry: Dict) -> Optional[str]:
        """Extract audio URL from RSS entry."""
        if 'enclosures' in entry:
            for enclosure in entry.enclosures:
                if 'audio' in enclosure.get('type', '').lower():
                    return enclosure.get('href')
        
        if 'links' in entry:
            for link in entry.links:
                if 'audio' in link.get('type', '').lower():
                    return link.get('href')
        
        return None
    
    def _extract_episode_number(self, entry: Dict) -> Optional[str]:
        """Extract episode number from entry if available."""
        import re
        
        title = entry.get('title', '')
        patterns = [
            r'#(\d+)',
            r'Episode\s+(\d+)',
            r'Ep\.?\s*(\d+)',
            r'(\d+)\s*[:\-]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_duration(self, entry: Dict) -> Optional[int]:
        """Extract duration in seconds if available."""
        if 'itunes_duration' in entry:
            duration_str = entry.itunes_duration
            try:
                parts = duration_str.split(':')
                if len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                else:
                    return int(parts[0])
            except:
                pass
        
        return None
    
    def _extract_youtube_urls(self, entry: Dict) -> Optional[List[str]]:
        """Extract YouTube URLs from episode description and content."""
        urls = []
        
        # Search in description/summary
        description = entry.get('summary', '') or entry.get('content', '')
        if description:
            if isinstance(description, list):
                description = ' '.join([str(item.get('value', '')) for item in description])
            urls.extend(find_youtube_urls(str(description)))
        
        # Search in entry content
        if 'content' in entry:
            for content_item in entry.content:
                urls.extend(find_youtube_urls(content_item.get('value', '')))
        
        # Search in links
        if 'links' in entry:
            for link in entry.links:
                href = link.get('href', '')
                if 'youtube.com' in href or 'youtu.be' in href:
                    urls.append(href)
        
        # Remove duplicates and return
        unique_urls = list(set(urls))
        if unique_urls:
            logger.info(f"Found YouTube URLs for episode '{entry.get('title', '')}': {unique_urls}")
        
        return unique_urls if unique_urls else None