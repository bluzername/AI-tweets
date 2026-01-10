"""
Lex Fridman Podcast scraper.

Most episodes have YouTube links in RSS - extract from there.
Episode numbers are in format "#NNN - Guest Name"
"""

import re
import logging
import requests
from typing import Optional
from bs4 import BeautifulSoup

from ..base import PodcastScraper

logger = logging.getLogger(__name__)


class LexFridmanScraper(PodcastScraper):
    """
    Scrapes thumbnails for Lex Fridman Podcast.

    Strategy:
    1. YouTube thumbnail from episode youtube_urls
    2. YouTube thumbnail from description links
    3. Open Graph image from lexfridman.com episode page
    """

    name = "lex_fridman"
    podcast_names = ["Lex Fridman Podcast"]
    BASE_URL = "https://lexfridman.com"

    def get_thumbnail_url(self, episode) -> Optional[str]:
        """Get thumbnail for Lex Fridman episode."""

        # Method 1: YouTube URL from episode data
        if hasattr(episode, 'youtube_urls') and episode.youtube_urls:
            urls = episode.youtube_urls if isinstance(episode.youtube_urls, list) else [episode.youtube_urls]
            for url in urls:
                video_id = self._extract_youtube_id(url)
                if video_id:
                    thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                    logger.debug(f"Lex Fridman: Found YouTube thumbnail from urls")
                    return thumbnail

        # Method 2: Find YouTube link in description
        if hasattr(episode, 'description') and episode.description:
            video_id = self._find_youtube_in_text(episode.description)
            if video_id:
                thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
                logger.debug(f"Lex Fridman: Found YouTube thumbnail from description")
                return thumbnail

        # Method 3: Scrape episode page from website
        episode_url = self._find_episode_page_url(episode)
        if episode_url:
            og_image = self._scrape_og_image(episode_url)
            if og_image:
                logger.debug(f"Lex Fridman: Found OG image from website")
                return og_image

        return None

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL."""
        if not url:
            return None
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _find_youtube_in_text(self, text: str) -> Optional[str]:
        """Find YouTube video ID in text."""
        return self._extract_youtube_id(text)

    def _find_episode_page_url(self, episode) -> Optional[str]:
        """Find episode page URL on lexfridman.com."""
        title = getattr(episode, 'title', '')
        if not title:
            return None

        # Episode number pattern: "#NNN - Guest Name" or "#NNN – Guest Name"
        match = re.match(r'#(\d+)', title)
        if match:
            episode_num = match.group(1)
            return f"{self.BASE_URL}/podcast/{episode_num}/"

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
