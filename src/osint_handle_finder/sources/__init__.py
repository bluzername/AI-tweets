"""
OSINT Handle Finder Sources.

Each source implements a different method for finding Twitter handles.
Sources are ordered by priority (lower = higher priority):
  - PodcastNotesSource (5): Direct context from episode metadata
  - TwitterAPISource (10): Official Twitter API lookup
  - WikidataSource (15): Structured data from Wikidata
  - DuckDuckGoSource (20): Web search results
  - LLMInferenceSource (50): AI-based inference (fallback)
"""

from .base import HandleSource
from .wikidata import WikidataSource
from .twitter_api import TwitterAPISource
from .web_search import DuckDuckGoSource
from .podcast_notes import PodcastNotesSource
from .llm_inference import LLMInferenceSource

# All available sources in priority order
ALL_SOURCES = [
    PodcastNotesSource,
    TwitterAPISource,
    WikidataSource,
    DuckDuckGoSource,
    LLMInferenceSource,
]

__all__ = [
    'HandleSource',
    'WikidataSource',
    'TwitterAPISource',
    'DuckDuckGoSource',
    'PodcastNotesSource',
    'LLMInferenceSource',
    'ALL_SOURCES',
]
