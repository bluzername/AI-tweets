"""Local Whisper transcription for unlimited file sizes and zero API costs."""

import logging
import os
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalWhisperTranscriber:
    """Handles transcription using local Whisper installation."""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize local Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (auto, cpu, cuda)
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._model_loaded = False
        
        logger.info(f"LocalWhisper configured: {model_size} model, {device} device")
    
    def _load_model(self):
        """Load Whisper model on first use."""
        if self._model_loaded:
            return
            
        try:
            import whisper
            logger.info(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            self._model_loaded = True
            logger.info("✅ Local Whisper model loaded successfully")
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
            
            logger.info(f"Transcribing {file_size:.1f}MB file with local Whisper...")
            
            # Transcribe with local Whisper
            options = {"fp16": False}  # More compatible
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
    
    def _download_audio(self, url: str) -> str:
        """Download audio file to temporary location."""
        import requests
        
        response = requests.get(url, stream=True, timeout=30)
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
                with open(cache_file, 'r') as f:
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
            with open(cache_file, 'w') as f:
                json.dump(transcription, f)
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