"""Audio transcription module using OpenAI Whisper API."""

import os
import logging
import tempfile
import time
import requests
from typing import Optional, Dict, Any
from pathlib import Path
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def retry_download(max_retries: int = 3, base_delay: float = 2.0):
    """Retry decorator for download operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Download retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


class WhisperTranscriber:
    """Handles audio transcription using OpenAI Whisper API."""
    
    def __init__(self, api_key: str, model: str = "whisper-1"):
        """
        Initialize Whisper transcriber.
        
        Args:
            api_key: OpenAI API key
            model: Whisper model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_file_size = 25 * 1024 * 1024  # 25MB API limit
        self.use_chunking = True  # Use native API chunking for large files
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def transcribe_audio(self, audio_url: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio from URL using Whisper API.
        
        Args:
            audio_url: URL of the audio file
            language: Optional language code for transcription
            
        Returns:
            Dictionary containing transcription and metadata
        """
        temp_file = None
        try:
            temp_file = self._download_audio(audio_url)
            
            with open(temp_file, 'rb') as audio_file:
                params = {
                    "model": self.model,
                    "response_format": "verbose_json"
                }
                
                if language:
                    params["language"] = language
                
                # Use native API chunking for large files
                file_size = os.path.getsize(temp_file)
                if file_size > self.max_file_size and self.use_chunking:
                    logger.info(f"Large file ({file_size / 1024 / 1024:.1f}MB), using API chunking")
                    params["chunking_strategy"] = {"type": "segment"}
                
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    **params
                )
                
                result = {
                    "text": response.text,
                    "segments": getattr(response, 'segments', []),
                    "language": getattr(response, 'language', language),
                    "duration": getattr(response, 'duration', None)
                }
                
                logger.info(f"Successfully transcribed audio: {audio_url[:50]}...")
                return result
                
        except Exception as e:
            logger.error(f"Error transcribing audio {audio_url}: {e}")
            raise
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
    
    @retry_download(max_retries=3, base_delay=2.0)
    def _download_audio(self, url: str) -> str:
        """
        Download audio file to temporary location with retry support.
        
        Args:
            url: URL of the audio file
            
        Returns:
            Path to temporary file
        """
        response = requests.get(url, stream=True, timeout=60)  # Increased timeout
        response.raise_for_status()
        
        suffix = self._get_file_extension(url, response.headers)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            
            logger.info(f"Downloaded audio to {temp_file.name}")
            return temp_file.name
    
    def _get_file_extension(self, url: str, headers: Dict) -> str:
        """Extract file extension from URL or headers."""
        content_type = headers.get('content-type', '')
        
        type_to_ext = {
            'audio/mpeg': '.mp3',
            'audio/mp4': '.m4a',
            'audio/wav': '.wav',
            'audio/x-wav': '.wav',
            'audio/webm': '.webm'
        }
        
        for mime_type, ext in type_to_ext.items():
            if mime_type in content_type:
                return ext
        
        from urllib.parse import urlparse
        path = urlparse(url).path
        if '.' in path:
            return '.' + path.split('.')[-1]
        
        return '.mp3'
    


class TranscriptionCache:
    """Simple cache for storing transcriptions."""
    
    def __init__(self, cache_dir: str = ".transcription_cache"):
        """Initialize cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get(self, audio_url: str) -> Optional[Dict[str, Any]]:
        """Get cached transcription if exists."""
        import hashlib
        import json
        
        cache_key = hashlib.md5(audio_url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"Using cached transcription for {audio_url[:50]}...")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        
        return None
    
    def set(self, audio_url: str, transcription: Dict[str, Any]):
        """Store transcription in cache."""
        import hashlib
        import json
        
        cache_key = hashlib.md5(audio_url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False)
                logger.info(f"Cached transcription for {audio_url[:50]}...")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")