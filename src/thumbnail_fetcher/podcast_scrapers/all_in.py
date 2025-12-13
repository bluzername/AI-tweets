"""
All-In Podcast scraper.

All-In episodes are primarily on YouTube, making thumbnail extraction reliable.
"""

import re
import logging
from typing import Optional

from ..base import PodcastScraper

logger = logging.getLogger(__name__)


class AllInScraper(PodcastScraper):
    """
    Scrapes thumbnails for All-In Podcast.

    Strategy: YouTube is the primary source for All-In
    """

    name = "all_in"
    podcast_names = [
        "All-In with Chamath, Jason, Sacks & Friedberg",
        "The All-In Podcast",
        "All-In Podcast"
    ]

    def get_thumbnail_url(self, episode) -> Optional[str]:
        """Get thumbnail for All-In episode."""

        # Method 1: YouTube URL from episode data
        if hasattr(episode, 'youtube_urls') and episode.youtube_urls:
            urls = episode.youtube_urls if isinstance(episode.youtube_urls, list) else [episode.youtube_urls]
            for url in urls:
                video_id = self._extract_youtube_id(url)
                if video_id:
                    logger.debug(f"All-In: Found YouTube thumbnail from urls")
                    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

        # Method 2: Find YouTube link in description
        if hasattr(episode, 'description') and episode.description:
            video_id = self._extract_youtube_id(episode.description)
            if video_id:
                logger.debug(f"All-In: Found YouTube thumbnail from description")
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
