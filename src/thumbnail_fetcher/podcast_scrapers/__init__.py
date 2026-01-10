"""
Podcast-specific scrapers for high-quality thumbnails.

These provide custom extraction for major podcasts that may
have better quality images on their websites than in RSS.
"""

from typing import Optional, Dict
from ..base import PodcastScraper

# Import all scrapers
from .lex_fridman import LexFridmanScraper
from .huberman_lab import HubermanLabScraper
from .a16z import A16ZScraper
from .all_in import AllInScraper
from .acquired import AcquiredScraper

# Registry of podcast scrapers
_SCRAPERS: Dict[str, PodcastScraper] = {}


def _register_scrapers():
    """Register all available scrapers."""
    scrapers = [
        LexFridmanScraper(),
        HubermanLabScraper(),
        A16ZScraper(),
        AllInScraper(),
        AcquiredScraper(),
    ]

    for scraper in scrapers:
        for name in scraper.podcast_names:
            _SCRAPERS[name] = scraper


# Initialize on import
_register_scrapers()


def get_scraper_for_podcast(podcast_name: str) -> Optional[PodcastScraper]:
    """
    Get scraper for a specific podcast, if available.

    Args:
        podcast_name: Name of the podcast

    Returns:
        PodcastScraper instance or None
    """
    return _SCRAPERS.get(podcast_name)


def get_all_scrapers() -> Dict[str, PodcastScraper]:
    """Get all registered scrapers."""
    return _SCRAPERS.copy()


__all__ = [
    'get_scraper_for_podcast',
    'get_all_scrapers',
    'LexFridmanScraper',
    'HubermanLabScraper',
    'A16ZScraper',
    'AllInScraper',
    'AcquiredScraper',
]
