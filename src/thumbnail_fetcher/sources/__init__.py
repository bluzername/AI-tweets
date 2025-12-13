"""
Thumbnail sources for multi-source fetching.
"""

from .youtube_source import YouTubeSource
from .rss_source import RSSSource
from .apple_source import ApplePodcastsSource

__all__ = ['YouTubeSource', 'RSSSource', 'ApplePodcastsSource']
