"""
YouTube thumbnail source - highest success rate for tech podcasts.

Most podcasts have YouTube versions with high-quality thumbnails.
YouTube thumbnail URLs are predictable and don't require API access.
"""

import re
import logging
import requests
from typing import Optional, List, Tuple

from ..base import ThumbnailSource

logger = logging.getLogger(__name__)


class YouTubeSource(ThumbnailSource):
    """
    Fetches thumbnails from YouTube.

    Strategy:
    1. Check episode.youtube_urls (already extracted during ingestion)
    2. Search episode description for YouTube links
    3. Try highest quality first, fall back to lower quality

    YouTube thumbnail URL format:
    https://img.youtube.com/vi/{video_id}/{quality}.jpg

    Quality levels:
    - maxresdefault (1280x720) - may not exist for all videos
    - sddefault (640x480)
    - hqdefault (480x360) - always exists
    - mqdefault (320x180)
    """

    name = "youtube"
    priority = 2  # High priority (after RSS)

    # Quality levels to try (best to worst)
    QUALITY_LEVELS: List[Tuple[str, int, int]] = [
        ("maxresdefault", 1280, 720),   # HD - may not exist
        ("sddefault", 640, 480),         # SD
        ("hqdefault", 480, 360),         # High quality - always exists
    ]

    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]

    def get_thumbnail_url(self, episode) -> Optional[str]:
        """Get YouTube thumbnail URL for episode."""
        video_id = self._find_video_id(episode)
        if not video_id:
            logger.debug(f"No YouTube video ID found for: {episode.title[:50]}")
            return None

        thumbnail_url = self._get_best_quality(video_id)
        if thumbnail_url:
            logger.debug(f"Found YouTube thumbnail for: {episode.title[:50]}")
        return thumbnail_url

    def _find_video_id(self, episode) -> Optional[str]:
        """Find YouTube video ID from episode data."""

        # Method 1: Use youtube_urls from episode (already parsed during ingestion)
        if hasattr(episode, 'youtube_urls') and episode.youtube_urls:
            urls = episode.youtube_urls
            if isinstance(urls, str):
                urls = [urls]
            for url in urls:
                video_id = self._extract_youtube_id(url)
                if video_id:
                    logger.debug(f"Found video ID from youtube_urls: {video_id}")
                    return video_id

        # Method 2: Search description for YouTube links
        if hasattr(episode, 'description') and episode.description:
            video_id = self._find_youtube_in_text(episode.description)
            if video_id:
                logger.debug(f"Found video ID in description: {video_id}")
                return video_id

        return None

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        if not url:
            return None
        for pattern in self.YOUTUBE_PATTERNS:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _find_youtube_in_text(self, text: str) -> Optional[str]:
        """Find first YouTube video ID in text."""
        if not text:
            return None
        for pattern in self.YOUTUBE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _get_best_quality(self, video_id: str) -> Optional[str]:
        """
        Find highest quality thumbnail that actually exists.

        Some videos don't have maxresdefault - need to check.
        """
        for quality, width, height in self.QUALITY_LEVELS:
            url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"

            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    # Check content-length to verify it's not a placeholder
                    content_length = int(response.headers.get('content-length', 0))
                    if content_length > 1000:  # Real thumbnails are > 1KB
                        logger.debug(f"Found {quality} thumbnail ({width}x{height}) for {video_id}")
                        return url
            except requests.RequestException as e:
                logger.debug(f"Failed to check {quality} for {video_id}: {e}")
                continue

        # Fallback to hqdefault which always exists
        fallback = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        logger.debug(f"Using fallback hqdefault for {video_id}")
        return fallback

    def verify_url(self, url: str) -> bool:
        """Verify that a thumbnail URL is valid and accessible."""
        try:
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
