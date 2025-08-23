"""YouTube transcript extraction module for cost-effective transcription."""

import logging
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter

logger = logging.getLogger(__name__)


class YouTubeTranscriber:
    """Handles transcript extraction from YouTube videos."""
    
    def __init__(self):
        """Initialize YouTube transcriber."""
        self.formatter = TextFormatter()
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract YouTube video ID from various URL formats.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID or None if not found
        """
        if not url:
            return None
            
        # Common YouTube URL patterns (YouTube video IDs are typically 11 characters but can vary)
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]+)',
            r'youtube\.com/v/([a-zA-Z0-9_-]+)',
            r'youtube\.com/watch\?.*?v=([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Try parsing as URL parameters
        try:
            parsed = urlparse(url)
            if 'youtube.com' in parsed.netloc:
                query_params = parse_qs(parsed.query)
                if 'v' in query_params:
                    return query_params['v'][0]
        except Exception as e:
            logger.debug(f"Error parsing YouTube URL {url}: {e}")
        
        return None
    
    def get_transcript(self, video_id: str, languages: List[str] = None) -> Dict[str, Any]:
        """
        Get transcript for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages (defaults to ['en'])
            
        Returns:
            Dictionary containing transcript text and metadata
        """
        if not languages:
            languages = ['en']
        
        try:
            # Use the simpler API - just fetch transcript directly
            transcript_data = YouTubeTranscriptApi().fetch(video_id, languages)
            
            # Format as plain text
            text = self.formatter.format_transcript(transcript_data.transcript)
            
            # Calculate approximate duration from last timestamp
            duration = None
            if transcript_data.transcript:
                last_segment = max(transcript_data.transcript, key=lambda x: x.get('start', 0))
                duration = last_segment.get('start', 0) + last_segment.get('duration', 0)
            
            result = {
                'text': text,
                'segments': transcript_data.transcript,
                'language': transcript_data.language_code,
                'language_name': getattr(transcript_data, 'language', transcript_data.language_code),
                'duration': duration,
                'transcript_type': "auto" if transcript_data.is_generated else "manual",
                'source': 'youtube',
                'video_id': video_id
            }
            
            logger.info(f"Successfully extracted transcript for video {video_id} "
                       f"in {transcript_data.language_code}")
            
            return result
            
        except TranscriptsDisabled:
            logger.warning(f"Transcripts disabled for video {video_id}")
            raise
        except NoTranscriptFound:
            logger.warning(f"No transcript found for video {video_id}")
            raise
        except Exception as e:
            logger.error(f"Error fetching transcript for video {video_id}: {e}")
            raise
    
    def transcribe_from_url(self, youtube_url: str, languages: List[str] = None) -> Dict[str, Any]:
        """
        Extract transcript from YouTube URL.
        
        Args:
            youtube_url: YouTube video URL
            languages: Preferred languages
            
        Returns:
            Transcript data
        """
        video_id = self.extract_video_id(youtube_url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {youtube_url}")
        
        return self.get_transcript(video_id, languages)
    
    def get_available_transcripts(self, video_id: str) -> List[Dict[str, str]]:
        """
        Get list of available transcripts for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of available transcripts with metadata
        """
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            
            available = []
            for transcript in transcript_list:
                available.append({
                    'language_code': transcript.language_code,
                    'language_name': getattr(transcript, 'language', transcript.language_code),
                    'is_generated': transcript.is_generated,
                    'is_translatable': getattr(transcript, 'is_translatable', False)
                })
            
            return available
            
        except Exception as e:
            logger.error(f"Error getting available transcripts for {video_id}: {e}")
            return []


class YouTubeTranscriptCache:
    """Cache for YouTube transcripts to avoid repeated API calls."""
    
    def __init__(self, cache_dir: str = ".youtube_transcript_cache"):
        """Initialize cache."""
        from pathlib import Path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get cached transcript if exists."""
        import json
        
        cache_file = self.cache_dir / f"{video_id}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"Using cached YouTube transcript for video {video_id}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading YouTube transcript cache: {e}")
        
        return None
    
    def set(self, video_id: str, transcript_data: Dict[str, Any]):
        """Store transcript in cache."""
        import json
        
        cache_file = self.cache_dir / f"{video_id}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Cached YouTube transcript for video {video_id}")
        except Exception as e:
            logger.error(f"Error writing YouTube transcript cache: {e}")


def find_youtube_urls(text: str) -> List[str]:
    """
    Find all YouTube URLs in a text string.
    
    Args:
        text: Text to search for YouTube URLs
        
    Returns:
        List of YouTube URLs found
    """
    if not text:
        return []
    
    patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?[^\s]+',
        r'https?://youtu\.be/[^\s]+',
        r'https?://(?:www\.)?youtube\.com/embed/[^\s]+',
        r'https?://(?:www\.)?youtube\.com/v/[^\s]+'
    ]
    
    urls = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        urls.extend(matches)
    
    return list(set(urls))  # Remove duplicates