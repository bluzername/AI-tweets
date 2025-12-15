"""Local Whisper transcription for unlimited file sizes and zero API costs."""

import logging
import os
import tempfile
import time
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def retry_on_network_error(max_retries: int = 5, base_delay: float = 3.0):
    """
    Retry decorator for network operations with exponential backoff.

    Handles HTTP 500 errors (server issues) and network errors with increasing delays.
    Default: up to 5 retries with delays of 3s, 6s, 12s, 24s, 48s (total ~93s max wait).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    # Retry on network errors AND server errors (500, 502, 503, 504)
                    is_retriable = any(kw in error_str for kw in [
                        'timeout', 'connection', 'network', 'ssl', 'socket',
                        'temporary', 'unavailable', '500', '502', '503', '504',
                        'server error', 'internal server error'
                    ])
                    if is_retriable and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                                     f"(waiting {delay:.1f}s): {e}")
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


class LocalWhisperTranscriber:
    """Handles transcription using local Whisper installation."""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize local Whisper transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (auto, cpu, cuda, mps)
        """
        self.model_size = model_size

        # Auto-detect best device for Apple Silicon
        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
                logger.info("🚀 Detected Apple Silicon GPU (M3 Max) - Using Metal Performance Shaders (MPS)")
            elif torch.cuda.is_available():
                self.device = "cuda"  # NVIDIA GPU
                logger.info("🚀 Detected CUDA GPU")
            else:
                self.device = "cpu"
                logger.info("Using CPU (no GPU detected)")
        else:
            self.device = device

        self.model = None
        self._model_loaded = False

        logger.info(f"LocalWhisper configured: {model_size} model, {self.device} device")
    
    def _load_model(self):
        """Load Whisper model on first use with MPS fallback to CPU."""
        if self._model_loaded:
            return

        try:
            import whisper
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")

            try:
                self.model = whisper.load_model(self.model_size, device=self.device)
            except (RuntimeError, Exception) as e:
                error_str = str(e).lower()
                # Catch MPS-related errors: sparse tensor ops, storage issues, Metal errors
                mps_error_keywords = ["mps", "storage", "metal", "sparse"]
                if self.device == "mps" and any(kw in error_str for kw in mps_error_keywords):
                    logger.warning(f"MPS device error, falling back to CPU: {str(e)[:200]}")
                    self.device = "cpu"
                    self.model = whisper.load_model(self.model_size, device=self.device)
                else:
                    raise

            self._model_loaded = True
            logger.info(f"✅ Local Whisper model loaded successfully on {self.device.upper()}")
        except ImportError:
            raise ImportError(
                "Local Whisper not installed. Run: pip install openai-whisper\n"
                "Note: This requires ~2GB disk space and may take time to download models."
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_url: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio using local Whisper.
        
        Args:
            audio_url: URL of the audio file
            language: Optional language code
            
        Returns:
            Dictionary containing transcription and metadata
        """
        self._load_model()
        
        temp_file = None
        try:
            temp_file = self._download_audio(audio_url)
            file_size = os.path.getsize(temp_file) / 1024 / 1024  # MB
            
            logger.info(f"Transcribing {file_size:.1f}MB file with local Whisper on {self.device.upper()}...")

            # Transcribe with local Whisper
            # Disable fp16 on all devices - fp16 causes NaN errors on some NVIDIA GPUs
            # The speed difference is minimal and fp32 is more stable
            options = {"fp16": False}
            if language:
                options["language"] = language

            result = self.model.transcribe(temp_file, **options)
            
            # Format to match API response
            formatted_result = {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", language),
                "duration": result.get("duration"),
                "source": "local_whisper",
                "model_size": self.model_size,
                "file_size_mb": file_size
            }
            
            logger.info(f"✅ Local transcription complete: {len(result['text'])} chars, "
                       f"{file_size:.1f}MB file, COST: $0.00")
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            raise
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
    
    @retry_on_network_error(max_retries=5, base_delay=5.0)
    def _download_audio(self, url: str) -> str:
        """Download audio file to temporary location with retry support.

        Uses exponential backoff for HTTP 500 errors (common with Flightcast CDN).
        Max wait time: 5s + 10s + 20s + 40s + 80s = ~155s total.
        """
        import requests

        response = requests.get(url, stream=True, timeout=90)  # Longer timeout for slow CDNs
        response.raise_for_status()
        
        suffix = self._get_file_extension(url, response.headers)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "loaded": self._model_loaded,
            "supports_unlimited_files": True,
            "cost_per_minute": 0.0,
            "quality": "Same as OpenAI API"
        }


class LocalWhisperCache:
    """Cache for local Whisper transcriptions."""
    
    def __init__(self, cache_dir: str = ".local_whisper_cache"):
        """Initialize cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get(self, audio_url: str, model_size: str) -> Optional[Dict[str, Any]]:
        """Get cached transcription if exists."""
        import hashlib
        import json
        
        cache_key = hashlib.md5(f"{audio_url}_{model_size}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"Using cached local Whisper transcription")
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading local Whisper cache: {e}")
        
        return None
    
    def set(self, audio_url: str, model_size: str, transcription: Dict[str, Any]):
        """Store transcription in cache."""
        import hashlib
        import json
        
        cache_key = hashlib.md5(f"{audio_url}_{model_size}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, ensure_ascii=False)
                logger.info(f"Cached local Whisper transcription")
        except Exception as e:
            logger.error(f"Error writing local Whisper cache: {e}")


def is_whisper_available() -> bool:
    """Check if local Whisper is available."""
    try:
        import whisper
        return True
    except ImportError:
        return False


def get_recommended_model_size(file_size_mb: float) -> str:
    """Get recommended Whisper model size based on file size and performance needs."""
    if file_size_mb < 10:
        return "base"      # Fast, good quality for small files
    elif file_size_mb < 50:
        return "small"     # Balanced speed/quality
    elif file_size_mb < 100:
        return "medium"    # Better quality for longer content
    else:
        return "large"     # Best quality for very long content


def estimate_transcription_time(file_size_mb: float, model_size: str = "base") -> float:
    """Estimate transcription time in minutes."""
    # Rough estimates based on typical CPU performance
    base_time_per_mb = {
        "tiny": 0.1,
        "base": 0.3, 
        "small": 0.5,
        "medium": 1.0,
        "large": 2.0
    }
    
    return file_size_mb * base_time_per_mb.get(model_size, 0.3)