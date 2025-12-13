"""
RSS feed thumbnail source.

Extracts thumbnails from RSS feed metadata that was captured during ingestion.
This is the original source - kept for podcasts like DOAC that have good RSS images.
"""

import logging
from typing import Optional

from ..base import ThumbnailSource

logger = logging.getLogger(__name__)


class RSSSource(ThumbnailSource):
    """
    Uses thumbnail URL already extracted from RSS feed during ingestion.

    This source simply returns the thumbnail_url that was stored
    in the episode data when the RSS feed was parsed.

    Priority: Highest (if available, RSS images are authoritative)
    """

    name = "rss"
    priority = 1  # Highest priority

    def get_thumbnail_url(self, episode) -> Optional[str]:
        """Get thumbnail URL from RSS feed metadata."""
        # Check if episode already has a thumbnail URL from RSS parsing
        if hasattr(episode, 'thumbnail_url') and episode.thumbnail_url:
            url = episode.thumbnail_url
            # Validate it looks like a real URL
            if url.startswith(('http://', 'https://')):
                logger.debug(f"Found RSS thumbnail for: {episode.title[:50]}")
                return url

        return None
