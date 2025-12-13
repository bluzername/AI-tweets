"""
OSINT Handle Finder - Advanced Twitter handle discovery using OSINT techniques.

This module provides a robust system for finding Twitter/X.com handles
for podcast guests and hosts using multiple data sources:

- Wikidata SPARQL queries (P2002 property)
- Twitter API v2 (user lookup and search)
- DuckDuckGo web search
- Podcast episode notes parsing
- LLM-based inference

Features:
- Multi-source confidence scoring
- Verification layer
- Human review queue for uncertain matches
- SQLite caching (positive and negative)
- Parallel lookups for performance

Usage:
    from osint_handle_finder import OSINTHandleFinder, HandleLookupContext

    finder = OSINTHandleFinder()
    result = finder.find_handle(
        "Naval Ravikant",
        context=HandleLookupContext(
            podcast_name="Tim Ferriss Show",
            known_profession="investor"
        )
    )

    if result.handle:
        print(f"Found: @{result.handle} ({result.confidence:.0%} confidence)")

CLI Usage:
    python -m osint_handle_finder.cli lookup "Elon Musk"
    python -m osint_handle_finder.cli review --list
    python -m osint_handle_finder.cli stats
"""

from .models import (
    HandleCandidate,
    HandleLookupResult,
    HandleLookupContext,
    LookupStatus,
    VerificationResult,
    ReviewQueueItem
)
from .handle_finder import OSINTHandleFinder, find_twitter_handle
from .database import HandleDatabase
from .confidence import ConfidenceScorer
from .verification import HandleVerifier
from .sources import (
    HandleSource,
    WikidataSource,
    TwitterAPISource,
    DuckDuckGoSource,
    PodcastNotesSource,
    LLMInferenceSource,
    ALL_SOURCES
)

__version__ = "1.0.0"
__author__ = "AI-Tweets Pipeline"

__all__ = [
    # Main classes
    'OSINTHandleFinder',
    'HandleDatabase',
    'ConfidenceScorer',
    'HandleVerifier',

    # Models
    'HandleCandidate',
    'HandleLookupResult',
    'HandleLookupContext',
    'LookupStatus',
    'VerificationResult',
    'ReviewQueueItem',

    # Sources
    'HandleSource',
    'WikidataSource',
    'TwitterAPISource',
    'DuckDuckGoSource',
    'PodcastNotesSource',
    'LLMInferenceSource',
    'ALL_SOURCES',

    # Convenience functions
    'find_twitter_handle',
]
