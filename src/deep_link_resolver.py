#!/usr/bin/env python3
"""
Deep Link Resolver - Get episode-specific Spotify and Apple Podcasts URLs.
"""

import requests
import logging
from typing import Optional, Dict
from urllib.parse import quote

logger = logging.getLogger(__name__)


class DeepLinkResolver:
    """Resolve episode-specific deep links for Spotify and Apple Podcasts."""

    def __init__(self, platform_config: Dict):
        """
        Initialize with platform configuration.

        Args:
            platform_config: Dict mapping podcast names to their platform IDs
                e.g., {"Lex Fridman Podcast": {"spotify_show_id": "...", "apple_podcast_id": "..."}}
        """
        self.config = platform_config or {}

    def get_episode_links(self, podcast_name: str, episode_title: str) -> Dict[str, str]:
        """
        Get deep links for a specific episode.

        Args:
            podcast_name: Name of the podcast (must match config key)
            episode_title: Title of the episode to find

        Returns:
            Dict with 'spotify' and/or 'apple' URLs if found
        """
        links = {}

        # Try to find config - check exact match first, then partial match
        podcast_config = self._find_podcast_config(podcast_name)

        if not podcast_config:
            logger.debug(f"No platform config found for: {podcast_name}")
            return links

        # Spotify: Construct show URL (Spotify doesn't have free episode search API)
        if spotify_id := podcast_config.get('spotify_show_id'):
            spotify_url = self._get_spotify_episode_url(spotify_id, episode_title)
            if spotify_url:
                links['spotify'] = spotify_url

        # Apple: Use iTunes Search API to find exact episode
        if apple_id := podcast_config.get('apple_podcast_id'):
            apple_url = self._get_apple_episode_url(apple_id, episode_title)
            if apple_url:
                links['apple'] = apple_url

        return links

    def _find_podcast_config(self, podcast_name: str) -> Optional[Dict]:
        """Find config for podcast, supporting partial name matches."""
        # Exact match
        if podcast_name in self.config:
            return self.config[podcast_name]

        # Partial match (case-insensitive)
        podcast_lower = podcast_name.lower()
        for key, value in self.config.items():
            if key.lower() in podcast_lower or podcast_lower in key.lower():
                return value

        return None

    def _get_spotify_episode_url(self, show_id: str, episode_title: str) -> Optional[str]:
        """
        Get Spotify URL for an episode.

        Note: Spotify's free API doesn't support episode search.
        We return the show page - users can find the episode there.
        For better results, would need Spotify Web API with auth.
        """
        try:
            # Direct link to show page
            return f"https://open.spotify.com/show/{show_id}"
        except Exception as e:
            logger.warning(f"Spotify link construction failed: {e}")
            return None

    def _get_apple_episode_url(self, podcast_id: str, episode_title: str) -> Optional[str]:
        """
        Search Apple Podcasts for specific episode URL.

        Uses iTunes Search API to find episodes by podcast ID,
        then fuzzy matches on episode title.
        """
        try:
            # iTunes API: Get episodes for this podcast
            search_url = (
                f"https://itunes.apple.com/lookup?"
                f"id={podcast_id}&entity=podcastEpisode&limit=100"
            )

            resp = requests.get(search_url, timeout=15)
            if not resp.ok:
                logger.warning(f"Apple API error: {resp.status_code}")
                return None

            data = resp.json()
            results = data.get('results', [])

            # First result is the podcast itself, rest are episodes
            for result in results:
                if result.get('wrapperType') != 'podcastEpisode':
                    continue

                track_name = result.get('trackName', '')

                # Try to match episode title
                if self._title_matches(episode_title, track_name):
                    episode_url = result.get('trackViewUrl')
                    if episode_url:
                        logger.debug(f"Found Apple episode: {track_name}")
                        return episode_url

            # Fallback: return podcast page if no episode match
            for result in results:
                if result.get('wrapperType') == 'podcast':
                    return result.get('collectionViewUrl')

            logger.debug(f"No Apple episode match for: {episode_title[:50]}")
            return None

        except requests.Timeout:
            logger.warning("Apple API timeout")
            return None
        except Exception as e:
            logger.warning(f"Apple episode search failed: {e}")
            return None

    def _title_matches(self, query: str, candidate: str) -> bool:
        """
        Fuzzy match episode titles.

        Checks if significant words from query appear in candidate.
        Handles episode numbering, guest names, etc.
        """
        if not query or not candidate:
            return False

        query_lower = query.lower()
        candidate_lower = candidate.lower()

        # Check for episode number match (e.g., "#470", "E470", "Episode 470")
        import re
        query_nums = set(re.findall(r'\d{2,4}', query_lower))
        candidate_nums = set(re.findall(r'\d{2,4}', candidate_lower))
        if query_nums and query_nums & candidate_nums:
            return True

        # Extract significant words (3+ chars, skip common words)
        skip_words = {'the', 'and', 'with', 'for', 'from', 'this', 'that', 'podcast', 'episode'}

        def get_words(text):
            words = set(w for w in re.findall(r'\w+', text.lower()) if len(w) > 2)
            return words - skip_words

        query_words = get_words(query)
        candidate_words = get_words(candidate)

        if not query_words:
            return False

        # Calculate overlap
        overlap = len(query_words & candidate_words)
        required = min(3, max(1, len(query_words) // 2))

        return overlap >= required


def format_listen_links(links: Dict[str, str], language: str = "en") -> str:
    """
    Format platform deep links for Telegram message.

    Args:
        links: Dict with 'spotify' and/or 'apple' URLs
        language: 'en' for English, 'he' for Hebrew

    Returns:
        Formatted markdown string for Telegram
    """
    if not links:
        return ""

    parts = []
    if 'spotify' in links:
        parts.append(f"[Spotify]({links['spotify']})")
    if 'apple' in links:
        parts.append(f"[Apple]({links['apple']})")

    if not parts:
        return ""

    if language == "he":
        return f"\n\n🎧 האזנה: {' | '.join(parts)}"
    else:
        return f"\n\n🎧 Listen: {' | '.join(parts)}"
