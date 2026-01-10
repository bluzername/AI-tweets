"""
Base classes for thumbnail sources and scrapers.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThumbnailResult:
    """Result from a thumbnail fetch attempt."""
    url: Optional[str] = None
    source: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ThumbnailSource(ABC):
    """
    Abstract base class for thumbnail sources.

    Each source implements a specific strategy for finding
    episode thumbnails (YouTube, RSS, Apple Podcasts, etc.)
    """

    name: str = "base"
    priority: int = 100  # Lower = higher priority

    @abstractmethod
    def get_thumbnail_url(self, episode) -> Optional[str]:
        """
        Get thumbnail URL for an episode.

        Args:
            episode: PodcastEpisode object with metadata

        Returns:
            URL string or None if not found
        """
        pass

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _find_youtube_in_text(self, text: str) -> Optional[str]:
        """Find first YouTube video ID in text."""
        if not text:
            return None
        return self._extract_youtube_id(text)


class PodcastScraper(ABC):
    """
    Abstract base class for podcast-specific scrapers.

    These provide custom thumbnail extraction for major podcasts
    that may have better quality images on their websites.
    """

    name: str = "base"
    podcast_names: List[str] = []  # Podcast names this scraper handles

    @abstractmethod
    def get_thumbnail_url(self, episode) -> Optional[str]:
        """
        Get thumbnail URL for an episode from this podcast.

        Args:
            episode: PodcastEpisode object

        Returns:
            URL string or None if not found
        """
        pass

    def supports_podcast(self, podcast_name: str) -> bool:
        """Check if this scraper supports the given podcast."""
        return podcast_name in self.podcast_names
