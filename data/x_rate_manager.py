#!/usr/bin/env python3
"""
X Rate Limit Manager - Robust rate limiting for X/Twitter API.
Designed for Free tier: 1,500 tweets/month, strict 15-min windows.
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Callable
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class XRateLimitConfig:
    """Configuration for X API rate limits."""
    # Free tier limits
    tweets_per_month: int = 1500
    tweets_per_15min_window: int = 15  # Conservative for free tier
    
    # Spacing settings
    min_seconds_between_tweets: float = 5.0  # Within a thread
    min_seconds_between_threads: float = 180.0  # 3 minutes between threads
    
    # Retry settings - use HOURS for X free tier rate limits
    max_retries: int = 5
    base_retry_delay: float = 1800.0  # 30 minutes base
    max_retry_delay: float = 14400.0  # 4 hours max
    
    # Safety buffer
    monthly_safety_buffer: float = 0.9  # Use only 90% of monthly limit


class XRateLimitManager:
    """
    Intelligent rate limit manager for X/Twitter API.
    
    Features:
    - Proactive rate limit tracking (no blocking waits)
    - 15-minute window management
    - Monthly quota tracking
    - Smart spacing between posts
    - Retry queue with exponential backoff
    - Persistence across restarts
    """
    
    def __init__(self, config: Optional[XRateLimitConfig] = None,
                 state_file: str = "data/x_rate_state.json"):
        self.config = config or XRateLimitConfig()
        self.state_file = Path(state_file)
        
        # Tweet history for window tracking (timestamps)
        self.tweet_timestamps: deque = deque(maxlen=1000)
        
        # Monthly tracking
        self.monthly_count: int = 0
        self.month_start: datetime = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        
        # Last post time
        self.last_tweet_time: float = 0
        self.last_thread_time: float = 0
        
        # Retry queue: list of (thread_data, retry_count, next_retry_time)
        self.retry_queue: List[tuple] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load persisted state
        self._load_state()
        
        logger.info(f"XRateLimitManager initialized: {self.config.tweets_per_15min_window}/15min, "
                   f"{self.config.tweets_per_month}/month")
    
    def _load_state(self):
        """Load persisted state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.monthly_count = state.get('monthly_count', 0)
                month_start_str = state.get('month_start')
                if month_start_str:
                    self.month_start = datetime.fromisoformat(month_start_str)
                
                # Reset if new month
                if datetime.now().month != self.month_start.month:
                    self.monthly_count = 0
                    self.month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0)
                
                self.last_tweet_time = state.get('last_tweet_time', 0)
                self.last_thread_time = state.get('last_thread_time', 0)
                
                logger.info(f"Loaded rate limit state: {self.monthly_count} tweets this month")
        except Exception as e:
            logger.warning(f"Could not load rate limit state: {e}")
    
    def _save_state(self):
        """Persist state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'monthly_count': self.monthly_count,
                'month_start': self.month_start.isoformat(),
                'last_tweet_time': self.last_tweet_time,
                'last_thread_time': self.last_thread_time,
                'updated_at': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save rate limit state: {e}")
    
    def _clean_old_timestamps(self):
        """Remove timestamps older than 15 minutes."""
        cutoff = time.time() - 900  # 15 minutes
        while self.tweet_timestamps and self.tweet_timestamps[0] < cutoff:
            self.tweet_timestamps.popleft()
    
    def get_tweets_in_window(self) -> int:
        """Get number of tweets in current 15-minute window."""
        with self.lock:
            self._clean_old_timestamps()
            return len(self.tweet_timestamps)
    
    def get_remaining_in_window(self) -> int:
        """Get remaining tweets allowed in current window."""
        return max(0, self.config.tweets_per_15min_window - self.get_tweets_in_window())
    
    def get_remaining_monthly(self) -> int:
        """Get remaining tweets for the month."""
        limit = int(self.config.tweets_per_month * self.config.monthly_safety_buffer)
        return max(0, limit - self.monthly_count)
    
    def get_seconds_until_window_reset(self) -> float:
        """Get seconds until oldest tweet falls out of 15-min window."""
        with self.lock:
            self._clean_old_timestamps()
            if not self.tweet_timestamps:
                return 0
            oldest = self.tweet_timestamps[0]
            reset_time = oldest + 900  # 15 minutes from oldest
            return max(0, reset_time - time.time())
    
    def can_post_thread(self, tweet_count: int) -> tuple[bool, str, float]:
        """
        Check if a thread can be posted now.
        
        Returns:
            (can_post, reason, wait_seconds)
        """
        with self.lock:
            now = time.time()
            
            # Check monthly limit
            if self.monthly_count + tweet_count > self.config.tweets_per_month * self.config.monthly_safety_buffer:
                remaining = self.get_remaining_monthly()
                return False, f"Monthly limit reached ({self.monthly_count}/{self.config.tweets_per_month})", 0
            
            # Check 15-minute window
            self._clean_old_timestamps()
            in_window = len(self.tweet_timestamps)
            if in_window + tweet_count > self.config.tweets_per_15min_window:
                wait = self.get_seconds_until_window_reset()
                return False, f"Window limit ({in_window}/{self.config.tweets_per_15min_window})", wait
            
            # Check minimum time since last thread
            time_since_thread = now - self.last_thread_time
            if time_since_thread < self.config.min_seconds_between_threads:
                wait = self.config.min_seconds_between_threads - time_since_thread
                return False, f"Thread spacing ({time_since_thread:.0f}s < {self.config.min_seconds_between_threads}s)", wait
            
            return True, "OK", 0
    
    def record_tweet(self):
        """Record that a tweet was posted."""
        with self.lock:
            now = time.time()
            self.tweet_timestamps.append(now)
            self.last_tweet_time = now
            self.monthly_count += 1
            self._save_state()
    
    def record_thread_complete(self):
        """Record that a thread was completed."""
        with self.lock:
            self.last_thread_time = time.time()
            self._save_state()
    
    def get_tweet_delay(self) -> float:
        """Get delay needed before next tweet in a thread."""
        with self.lock:
            elapsed = time.time() - self.last_tweet_time
            needed = self.config.min_seconds_between_tweets - elapsed
            return max(0, needed)
    
    def wait_for_tweet(self):
        """Wait the appropriate time before posting next tweet."""
        delay = self.get_tweet_delay()
        if delay > 0:
            logger.debug(f"Waiting {delay:.1f}s before next tweet")
            time.sleep(delay)
    
    def add_to_retry_queue(self, thread_data: dict, retry_count: int = 0):
        """Add a failed thread to the retry queue."""
        delay = min(
            self.config.base_retry_delay * (2 ** retry_count),
            self.config.max_retry_delay
        )
        next_retry = time.time() + delay
        
        with self.lock:
            self.retry_queue.append((thread_data, retry_count + 1, next_retry))
            logger.info(f"Added thread to retry queue (attempt {retry_count + 1}, "
                       f"retry in {delay:.0f}s)")
    
    def get_ready_retries(self) -> List[tuple]:
        """Get threads ready to retry."""
        now = time.time()
        ready = []
        remaining = []
        
        with self.lock:
            for item in self.retry_queue:
                thread_data, retry_count, next_retry = item
                if retry_count > self.config.max_retries:
                    logger.warning(f"Thread exceeded max retries, dropping")
                    continue
                if next_retry <= now:
                    ready.append((thread_data, retry_count))
                else:
                    remaining.append(item)
            self.retry_queue = remaining
        
        return ready
    
    def get_status(self) -> dict:
        """Get current rate limit status."""
        return {
            'tweets_in_window': self.get_tweets_in_window(),
            'window_limit': self.config.tweets_per_15min_window,
            'remaining_in_window': self.get_remaining_in_window(),
            'seconds_until_reset': self.get_seconds_until_window_reset(),
            'monthly_count': self.monthly_count,
            'monthly_limit': self.config.tweets_per_month,
            'remaining_monthly': self.get_remaining_monthly(),
            'retry_queue_size': len(self.retry_queue),
            'last_tweet_ago': time.time() - self.last_tweet_time if self.last_tweet_time else None,
            'last_thread_ago': time.time() - self.last_thread_time if self.last_thread_time else None
        }


def post_thread_with_rate_limit(rate_manager: XRateLimitManager,
                                 post_func: Callable,
                                 thread_data: dict,
                                 tweets: List[str]) -> tuple[bool, List[str]]:
    """
    Post a thread with proper rate limiting.
    
    Args:
        rate_manager: XRateLimitManager instance
        post_func: Function to post a single tweet (text, reply_to_id) -> tweet_id
        thread_data: Thread metadata for retry queue
        tweets: List of tweet texts
    
    Returns:
        (success, list of tweet_ids)
    """
    # Check if we can post
    can_post, reason, wait_time = rate_manager.can_post_thread(len(tweets))
    
    if not can_post:
        if wait_time > 0 and wait_time < 300:  # Wait up to 5 minutes
            logger.info(f"Waiting {wait_time:.0f}s: {reason}")
            time.sleep(wait_time)
            # Re-check after waiting
            can_post, reason, wait_time = rate_manager.can_post_thread(len(tweets))
        
        if not can_post:
            logger.warning(f"Cannot post thread: {reason}")
            rate_manager.add_to_retry_queue(thread_data, 0)
            return False, []
    
    tweet_ids = []
    
    try:
        for i, tweet_text in enumerate(tweets):
            # Wait appropriate time between tweets
            rate_manager.wait_for_tweet()
            
            # Post the tweet
            reply_to = tweet_ids[-1] if tweet_ids else None
            tweet_id = post_func(tweet_text, reply_to)
            
            if tweet_id:
                tweet_ids.append(tweet_id)
                rate_manager.record_tweet()
                logger.info(f"Posted tweet {i+1}/{len(tweets)}: {tweet_id}")
            else:
                raise Exception(f"Failed to post tweet {i+1}")
        
        rate_manager.record_thread_complete()
        logger.info(f"Thread complete: {len(tweet_ids)} tweets")
        return True, tweet_ids
        
    except Exception as e:
        logger.error(f"Thread posting failed at tweet {len(tweet_ids)+1}: {e}")
        rate_manager.add_to_retry_queue(thread_data, 0)
        return False, tweet_ids


# Test/debug function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    mgr = XRateLimitManager()
    print("Rate Limit Status:")
    for k, v in mgr.get_status().items():
        print(f"  {k}: {v}")
