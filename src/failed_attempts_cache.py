"""Cache for failed transcription attempts to avoid repeated failures."""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FailedAttemptsCache:
    """Cache for failed transcription attempts."""
    
    def __init__(self, cache_dir: str = ".failed_attempts_cache", ttl_hours: int = 24):
        """
        Initialize failed attempts cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time to live for failed attempts (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_hours = ttl_hours
        
        logger.info(f"Failed attempts cache initialized: {cache_dir} (TTL: {ttl_hours}h)")
    
    def get(self, audio_url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached failed attempt if exists and not expired.
        
        Args:
            audio_url: Audio URL to check
            
        Returns:
            Failed attempt data if exists and not expired, None otherwise
        """
        cache_key = self._get_cache_key(audio_url)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if expired
            failed_time = datetime.fromisoformat(data['failed_at'])
            if datetime.now() - failed_time > timedelta(hours=self.ttl_hours):
                logger.debug(f"Failed attempt expired for {audio_url[:50]}...")
                self._remove(audio_url)
                return None
            
            logger.info(f"Using cached failed attempt for {audio_url[:50]}... (failed {failed_time.strftime('%Y-%m-%d %H:%M')})")
            return data
            
        except Exception as e:
            logger.error(f"Error reading failed attempts cache: {e}")
            return None
    
    def set(self, audio_url: str, method: str, error: str, attempts: list = None):
        """
        Cache a failed transcription attempt.
        
        Args:
            audio_url: Audio URL that failed
            method: Transcription method that failed
            error: Error message
            attempts: List of all attempts made
        """
        cache_key = self._get_cache_key(audio_url)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        data = {
            'audio_url': audio_url,
            'method': method,
            'error': error,
            'attempts': attempts or [],
            'failed_at': datetime.now().isoformat(),
            'ttl_hours': self.ttl_hours
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached failed attempt for {audio_url[:50]}... (method: {method})")
        except Exception as e:
            logger.error(f"Error writing failed attempts cache: {e}")
    
    def _remove(self, audio_url: str):
        """Remove a cached failed attempt."""
        cache_key = self._get_cache_key(audio_url)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            if cache_file.exists():
                cache_file.unlink()
                logger.debug(f"Removed expired failed attempt for {audio_url[:50]}...")
        except Exception as e:
            logger.error(f"Error removing failed attempt cache: {e}")
    
    def _get_cache_key(self, audio_url: str) -> str:
        """Generate cache key for audio URL."""
        return hashlib.md5(audio_url.encode()).hexdigest()
    
    def clear_expired(self):
        """Clear all expired failed attempts."""
        now = datetime.now()
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                failed_time = datetime.fromisoformat(data['failed_at'])
                if now - failed_time > timedelta(hours=self.ttl_hours):
                    cache_file.unlink()
                    cleared_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing cache file {cache_file}: {e}")
                # Remove corrupted cache files
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except:
                    pass
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} expired failed attempts")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_files = len(list(self.cache_dir.glob("*.json")))
        now = datetime.now()
        expired_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                failed_time = datetime.fromisoformat(data['failed_at'])
                if now - failed_time > timedelta(hours=self.ttl_hours):
                    expired_count += 1
            except:
                expired_count += 1
        
        return {
            'total_failed_attempts': total_files,
            'expired_attempts': expired_count,
            'active_attempts': total_files - expired_count,
            'cache_dir': str(self.cache_dir),
            'ttl_hours': self.ttl_hours
        }
