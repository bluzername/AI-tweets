"""
Apple Podcasts API source for thumbnails.

Free API, no authentication required.
Provides reliable fallback when other sources fail.
"""

import logging
import requests
from typing import Optional, Dict, List
from difflib import SequenceMatcher

from ..base import ThumbnailSource

logger = logging.getLogger(__name__)


class ApplePodcastsSource(ThumbnailSource):
    """
    Fetches thumbnails from Apple Podcasts.

    Strategy:
    1. Search for podcast by name
    2. Get episode list
    3. Match episode by title similarity
    4. Return episode artwork URL

    Note: Apple Podcasts API is free but may have rate limits.
    """

    name = "apple_podcasts"
    priority = 4  # Lower priority (fallback)

    SEARCH_URL = "https://itunes.apple.com/search"
    LOOKUP_URL = "https://itunes.apple.com/lookup"

    # Cache podcast IDs to avoid repeated searches
    _podcast_cache: Dict[str, int] = {}

    def get_thumbnail_url(self, episode) -> Optional[str]:
        """Get thumbnail from Apple Podcasts."""
        podcast_name = getattr(episode, 'podcast_name', '')
        if not podcast_name:
            return None

        # Find podcast ID
        podcast_id = self._get_podcast_id(podcast_name)
        if not podcast_id:
            logger.debug(f"Podcast not found on Apple: {podcast_name}")
            return None

        # Get episodes and match
        episodes = self._get_podcast_episodes(podcast_id)
        if not episodes:
            return None

        # Match episode by title
        matched = self._match_episode(episodes, episode)
        if matched:
            artwork_url = matched.get('artworkUrl600')
            if artwork_url:
                # Upgrade to higher resolution
                high_res = artwork_url.replace('600x600', '1400x1400')
                logger.debug(f"Found Apple Podcasts thumbnail for: {episode.title[:50]}")
                return high_res

        return None

    def _get_podcast_id(self, podcast_name: str) -> Optional[int]:
        """Search for podcast and get its ID."""
        if podcast_name in self._podcast_cache:
            return self._podcast_cache[podcast_name]

        try:
            response = requests.get(
                self.SEARCH_URL,
                params={
                    'term': podcast_name,
                    'media': 'podcast',
                    'entity': 'podcast',
                    'limit': 5
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            for result in data.get('results', []):
                # Check for close name match
                collection_name = result.get('collectionName', '')
                if self._similar(collection_name, podcast_name) > 0.7:
                    podcast_id = result.get('collectionId')
                    self._podcast_cache[podcast_name] = podcast_id
                    logger.debug(f"Found Apple podcast ID {podcast_id} for: {podcast_name}")
                    return podcast_id

        except Exception as e:
            logger.debug(f"Apple Podcasts search failed: {e}")

        return None

    def _get_podcast_episodes(self, podcast_id: int) -> List[Dict]:
        """Get recent episodes for podcast."""
        try:
            response = requests.get(
                self.LOOKUP_URL,
                params={
                    'id': podcast_id,
                    'entity': 'podcastEpisode',
                    'limit': 50
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Filter to only episode results (first result is the podcast itself)
            episodes = [
                r for r in data.get('results', [])
                if r.get('kind') == 'podcast-episode'
            ]
            return episodes

        except Exception as e:
            logger.debug(f"Failed to get Apple podcast episodes: {e}")
            return []

    def _match_episode(self, episodes: List[Dict], target) -> Optional[Dict]:
        """Find matching episode by title similarity."""
        target_title = getattr(target, 'title', '')
        if not target_title:
            return None

        best_match = None
        best_score = 0.0

        for ep in episodes:
            title = ep.get('trackName', '')
            score = self._similar(title, target_title)

            if score > best_score and score > 0.6:
                best_match = ep
                best_score = score

        if best_match:
            logger.debug(f"Matched episode with score {best_score:.2f}")

        return best_match

    @staticmethod
    def _similar(a: str, b: str) -> float:
        """Calculate string similarity ratio."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
