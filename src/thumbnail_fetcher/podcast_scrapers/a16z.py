"""
a16z Podcast scraper.

a16z has good episode thumbnails on their website.
"""

import re
import logging
import requests
from typing import Optional
from bs4 import BeautifulSoup

from ..base import PodcastScraper

logger = logging.getLogger(__name__)


class A16ZScraper(PodcastScraper):
    """
    Scrapes thumbnails for a16z podcasts.

    Strategy:
    1. YouTube thumbnail if available
    2. Scrape OG image from a16z.com episode page
    """

    name = "a16z"
    podcast_names = ["a16z Live", "a16z Podcast"]
    BASE_URL = "https://a16z.com"

    def get_thumbnail_url(self, episode) -> Optional[str]:
        """Get thumbnail for a16z episode."""

        # Method 1: YouTube URL from episode data
        if hasattr(episode, 'youtube_urls') and episode.youtube_urls:
            urls = episode.youtube_urls if isinstance(episode.youtube_urls, list) else [episode.youtube_urls]
            for url in urls:
                video_id = self._extract_youtube_id(url)
                if video_id:
                    logger.debug(f"a16z: Found YouTube thumbnail")
                    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

        # Method 2: Find YouTube link in description
        if hasattr(episode, 'description') and episode.description:
            video_id = self._extract_youtube_id(episode.description)
            if video_id:
                logger.debug(f"a16z: Found YouTube thumbnail from description")
                return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

            # Method 3: Find a16z.com link in description and scrape OG image
            a16z_url = self._find_a16z_url(episode.description)
            if a16z_url:
                og_image = self._scrape_og_image(a16z_url)
                if og_image:
                    logger.debug(f"a16z: Found OG image from website")
                    return og_image

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

    def _find_a16z_url(self, text: str) -> Optional[str]:
        """Find a16z.com URL in text."""
        if not text:
            return None
        # Look for a16z.com/podcast/ URLs
        match = re.search(r'https?://(?:www\.)?a16z\.com/podcast/[^\s<>"\']+', text, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    def _scrape_og_image(self, url: str) -> Optional[str]:
        """Scrape Open Graph image from webpage."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return og_image['content']
        except Exception as e:
            logger.debug(f"OG scraping failed for {url}: {e}")

        return None
