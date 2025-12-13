"""Multi-source transcriber supporting YouTube, Whisper, and other methods."""

import logging
import os
from typing import Optional, Dict, Any, List
from enum import Enum

from .youtube_transcriber import YouTubeTranscriber, YouTubeTranscriptCache
from .transcriber import WhisperTranscriber, TranscriptionCache
from .local_whisper import LocalWhisperTranscriber, LocalWhisperCache, is_whisper_available
from .failed_attempts_cache import FailedAttemptsCache

logger = logging.getLogger(__name__)


class TranscriptionMethod(Enum):
    """Supported transcription methods."""
    YOUTUBE = "youtube"
    WHISPER = "whisper" 
    LOCAL_WHISPER = "local_whisper"


class MultiTranscriber:
    """Handles transcription from multiple sources with cost optimization."""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 whisper_model: str = "whisper-1",
                 local_whisper_model: str = "base",
                 preferred_methods: List[str] = None):
        """
        Initialize multi-source transcriber.
        
        Args:
            openai_api_key: OpenAI API key for Whisper
            whisper_model: Whisper model to use
            local_whisper_model: Local Whisper model size (tiny, base, small, medium, large)
            preferred_methods: Ordered list of methods to try
        """
        self.openai_api_key = openai_api_key
        
        # Initialize transcribers
        self.youtube_transcriber = YouTubeTranscriber()
        self.youtube_cache = YouTubeTranscriptCache()
        
        if openai_api_key:
            self.whisper_transcriber = WhisperTranscriber(
                api_key=openai_api_key, 
                model=whisper_model
            )
            self.whisper_cache = TranscriptionCache()
        else:
            self.whisper_transcriber = None
            self.whisper_cache = None
        
        # Initialize local Whisper if available
        if is_whisper_available():
            # Use WHISPER_DEVICE env var (set by docker-compose.gpu.yml for GPU support)
            whisper_device = os.environ.get("WHISPER_DEVICE", "auto")
            self.local_whisper = LocalWhisperTranscriber(
                model_size=local_whisper_model,
                device=whisper_device
            )
            self.local_whisper_cache = LocalWhisperCache()
        else:
            self.local_whisper = None
            self.local_whisper_cache = None
        
        # Initialize failed attempts cache
        self.failed_attempts_cache = FailedAttemptsCache()
        
        # Set default method priority: free methods first, unlimited capacity second
        self.preferred_methods = preferred_methods or [
            TranscriptionMethod.YOUTUBE.value,
            TranscriptionMethod.LOCAL_WHISPER.value,
            TranscriptionMethod.WHISPER.value
        ]
        
        logger.info(f"Initialized MultiTranscriber with methods: {self.preferred_methods}")
        if self.local_whisper:
            logger.info(f"✅ Local Whisper available: {local_whisper_model} model (unlimited file size)")
        else:
            logger.info("⚠️  Local Whisper not available. Install with: pip install openai-whisper")
    
    def transcribe_episode(self, 
                          audio_url: str, 
                          youtube_urls: List[str] = None,
                          title: str = "") -> Dict[str, Any]:
        """
        Transcribe episode using the best available method.
        
        Args:
            audio_url: Direct audio URL
            youtube_urls: List of YouTube URLs if available
            title: Episode title for logging
            
        Returns:
            Transcription result with metadata
        """
        logger.info(f"Transcribing episode: {title}")
        
        # Check if we've already failed this audio URL recently
        failed_attempt = self.failed_attempts_cache.get(audio_url)
        if failed_attempt:
            logger.warning(f"⏭️  Skipping '{title}' - previous failure cached from {failed_attempt['failed_at']}")
            return {
                'text': '',
                'error': f'Cached failure from {failed_attempt["method"]}: {failed_attempt["error"]}',
                'attempts': failed_attempt['attempts'],
                'method_used': None,
                'cached_failure': True
            }
        
        # Track attempts and results
        attempts = []
        
        for method in self.preferred_methods:
            try:
                result = self._try_transcription_method(
                    method, audio_url, youtube_urls, title
                )
                if result:
                    result['method_used'] = method
                    result['attempts'] = attempts + [{'method': method, 'status': 'success'}]
                    
                    cost_info = self._get_cost_info(method, result)
                    logger.info(f"Successfully transcribed '{title}' using {method}. {cost_info}")
                    
                    return result
                    
            except Exception as e:
                attempts.append({'method': method, 'status': 'failed', 'error': str(e)})
                logger.warning(f"Failed to transcribe '{title}' using {method}: {e}")
                continue
        
        # All methods failed - cache the failure
        error_msg = f"All transcription methods failed for: {title}"
        logger.error(error_msg)
        
        # Cache the failure to avoid retrying
        self.failed_attempts_cache.set(
            audio_url=audio_url,
            method="all_methods",
            error=error_msg,
            attempts=attempts
        )
        
        return {
            'text': '',
            'error': error_msg,
            'attempts': attempts,
            'method_used': None
        }
    
    def _try_transcription_method(self, 
                                method: str, 
                                audio_url: str,
                                youtube_urls: List[str],
                                title: str) -> Optional[Dict[str, Any]]:
        """Try a specific transcription method."""
        
        if method == TranscriptionMethod.YOUTUBE.value:
            return self._try_youtube_transcription(youtube_urls, title)
        
        elif method == TranscriptionMethod.LOCAL_WHISPER.value:
            return self._try_local_whisper_transcription(audio_url, title)
        
        elif method == TranscriptionMethod.WHISPER.value:
            return self._try_whisper_transcription(audio_url, title)
        
        else:
            logger.warning(f"Unsupported transcription method: {method}")
            return None
    
    def _try_youtube_transcription(self, 
                                 youtube_urls: List[str],
                                 title: str) -> Optional[Dict[str, Any]]:
        """Try YouTube transcript extraction."""
        if not youtube_urls:
            logger.debug(f"No YouTube URLs available for: {title}")
            return None
        
        for url in youtube_urls:
            try:
                video_id = self.youtube_transcriber.extract_video_id(url)
                if not video_id:
                    continue
                
                # Check cache first
                cached = self.youtube_cache.get(video_id)
                if cached:
                    logger.info(f"Using cached YouTube transcript for: {title}")
                    return cached
                
                # Fetch transcript
                result = self.youtube_transcriber.transcribe_from_url(url)
                
                # Cache the result
                self.youtube_cache.set(video_id, result)
                
                return result
                
            except Exception as e:
                logger.debug(f"Failed to get YouTube transcript from {url}: {e}")
                continue
        
        logger.debug(f"No usable YouTube transcripts found for: {title}")
        return None
    
    def _try_local_whisper_transcription(self, 
                                       audio_url: str,
                                       title: str) -> Optional[Dict[str, Any]]:
        """Try local Whisper transcription."""
        if not self.local_whisper:
            logger.debug("Local Whisper not available (not installed)")
            return None
        
        try:
            # Check cache first
            cached = self.local_whisper_cache.get(audio_url, self.local_whisper.model_size)
            if cached:
                logger.info(f"Using cached local Whisper transcription for: {title}")
                return cached
            
            # Transcribe using local Whisper
            result = self.local_whisper.transcribe_audio(audio_url)
            
            # Cache the result
            if result:
                self.local_whisper_cache.set(audio_url, self.local_whisper.model_size, result)
            
            return result
            
        except Exception as e:
            logger.debug(f"Local Whisper transcription failed for {title}: {e}")
            return None
    
    def _try_whisper_transcription(self, 
                                 audio_url: str,
                                 title: str) -> Optional[Dict[str, Any]]:
        """Try Whisper API transcription."""
        if not self.whisper_transcriber:
            logger.debug("Whisper transcriber not available (no API key)")
            return None
        
        try:
            # Check cache first
            cached = self.whisper_cache.get(audio_url)
            if cached:
                logger.info(f"Using cached Whisper transcription for: {title}")
                return cached
            
            # Transcribe using Whisper
            result = self.whisper_transcriber.transcribe_audio(audio_url)
            
            # Cache the result
            if result:
                self.whisper_cache.set(audio_url, result)
            
            return result
            
        except Exception as e:
            logger.debug(f"Whisper transcription failed for {title}: {e}")
            return None
    
    def _get_cost_info(self, method: str, result: Dict[str, Any]) -> str:
        """Get cost information for logging."""
        if method == TranscriptionMethod.YOUTUBE.value:
            return "Cost: FREE"
        
        elif method == TranscriptionMethod.LOCAL_WHISPER.value:
            return "Cost: FREE (local processing)"
        
        elif method == TranscriptionMethod.WHISPER.value:
            duration = result.get('duration', 0)
            if duration:
                cost = duration / 60 * 0.006  # $0.006 per minute
                return f"Cost: ~${cost:.3f}"
            return "Cost: ~$0.006/min"
        
        return ""
    
    def get_transcription_stats(self) -> Dict[str, Any]:
        """Get statistics about transcription methods used."""
        # This could be enhanced to track usage statistics
        stats = {
            'methods_available': [],
            'youtube_enabled': True,
            'local_whisper_enabled': self.local_whisper is not None,
            'whisper_enabled': self.whisper_transcriber is not None,
            'preferred_order': self.preferred_methods
        }
        
        if True:  # YouTube always available
            stats['methods_available'].append('youtube')
        
        if self.local_whisper:
            stats['methods_available'].append('local_whisper')
        
        if self.whisper_transcriber:
            stats['methods_available'].append('whisper')
        
        return stats