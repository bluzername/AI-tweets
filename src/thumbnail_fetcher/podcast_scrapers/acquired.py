"""
Acquired podcast scraper.

Acquired has both YouTube videos and website episode pages.
"""

import re
import logging
import requests
from typing import Optional
from bs4 import BeautifulSoup

from ..base import PodcastScraper

logger = logging.getLogger(__name__)


class AcquiredScraper(PodcastScraper):
    """
    Scrapes thumbnails for Acquired podcast.

    Strategy:
    1. YouTube thumbnail from episode data
    2. acquired.fm episode page OG image
    """

    name = "acquired"
    podcast_names = ["Acquired"]
    BASE_URL = "https://www.acquired.fm"

    def get_thumbnail_url(self, episode) -> Optional[str]:
        """Get thumbnail for Acquired episode."""

        # Method 1: YouTube URL from episode data
        if hasattr(episode, 'youtube_urls') and episode.youtube_urls:
            urls = episode.youtube_urls if isinstance(episode.youtube_urls, list) else [episode.youtube_urls]
            for url in urls:
                video_id = self._extract_youtube_id(url)
                if video_id:
                    logger.debug(f"Acquired: Found YouTube thumbnail")
                    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

        # Method 2: Find YouTube link in description
        if hasattr(episode, 'description') and episode.description:
            video_id = self._extract_youtube_id(episode.description)
            if video_id:
                logger.debug(f"Acquired: Found YouTube thumbnail from description")
                return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

        return None

    def _extract_youtube_id(self, text: str) -> Optional[str]:
        """Extract YouTube video ID from text."""
        if not text:
            return None
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
