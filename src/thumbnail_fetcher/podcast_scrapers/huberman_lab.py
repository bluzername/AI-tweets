"""
Huberman Lab podcast scraper.

Uses YouTube as primary source since most episodes have video versions.
"""

import re
import logging
import requests
from typing import Optional
from bs4 import BeautifulSoup

from ..base import PodcastScraper

logger = logging.getLogger(__name__)


class HubermanLabScraper(PodcastScraper):
    """
    Scrapes thumbnails for Huberman Lab podcast.

    Strategy:
    1. YouTube thumbnail from episode youtube_urls
    2. YouTube thumbnail from description
    3. hubermanlab.com episode page OG image
    """

    name = "huberman_lab"
    podcast_names = ["Huberman Lab"]
    BASE_URL = "https://www.hubermanlab.com"

    def get_thumbnail_url(self, episode) -> Optional[str]:
        """Get thumbnail for Huberman Lab episode."""

        # Method 1: YouTube URL from episode data
        if hasattr(episode, 'youtube_urls') and episode.youtube_urls:
            urls = episode.youtube_urls if isinstance(episode.youtube_urls, list) else [episode.youtube_urls]
            for url in urls:
                video_id = self._extract_youtube_id(url)
                if video_id:
                    logger.debug(f"Huberman: Found YouTube thumbnail from urls")
                    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

        # Method 2: Find YouTube link in description
        if hasattr(episode, 'description') and episode.description:
            video_id = self._extract_youtube_id(episode.description)
            if video_id:
                logger.debug(f"Huberman: Found YouTube thumbnail from description")
                return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

        return None

    def _extract_youtube_id(self, text: str) -> Optional[str]:
        """Extract YouTube video ID from text."""
        if not text:
            return None
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
