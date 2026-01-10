#!/usr/bin/env python3
"""
Rate Limiter - Intelligent API rate limiting and throttling.
Prevents hitting API rate limits and optimizes request distribution.
"""

import logging
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    min_interval_seconds: float = 0.1  # Minimum time between requests


class RateLimiter:
    """
    Intelligent rate limiter with multiple time windows.

    Features:
    - Multiple time window tracking (minute, hour, day)
    - Automatic throttling
    - Request queuing
    - Burst handling
    - Per-service limits
    """

    # Default rate limits for various services
    DEFAULT_LIMITS = {
        "openai": RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=3500,
            requests_per_day=10000,
            min_interval_seconds=1.0
        ),
        "anthropic": RateLimitConfig(
            requests_per_minute=50,
            requests_per_hour=1000,
            requests_per_day=10000,
            min_interval_seconds=1.2
        ),
        "google": RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1500,
            requests_per_day=15000,
            min_interval_seconds=1.0
        ),
        "twitter": RateLimitConfig(
            requests_per_minute=15,
            requests_per_hour=100,
            requests_per_day=500,
            min_interval_seconds=4.0
        )
    }

    def __init__(self):
        """Initialize rate limiter."""
        self.limits = self.DEFAULT_LIMITS.copy()

        # Request history per service
        self.request_history: Dict[str, deque] = {}

        # Last request timestamp per service
        self.last_request: Dict[str, float] = {}

        # Locks for thread safety
        self.locks: Dict[str, threading.Lock] = {}

        # Initialize for each service
        for service in self.limits.keys():
            self.request_history[service] = deque()
            self.last_request[service] = 0.0
            self.locks[service] = threading.Lock()

    def set_limit(self, service: str, config: RateLimitConfig):
        """Set custom rate limit for a service."""
        self.limits[service] = config

        if service not in self.request_history:
            self.request_history[service] = deque()
            self.last_request[service] = 0.0
            self.locks[service] = threading.Lock()

        logger.info(f"Rate limit configured for {service}: {config.requests_per_minute} req/min")

    def _clean_old_requests(self, service: str):
        """Remove old requests outside tracking windows."""

        now = time.time()
        history = self.request_history[service]

        # Keep requests from last 24 hours
        cutoff = now - 86400  # 24 hours in seconds

        while history and history[0] < cutoff:
            history.popleft()

    def _get_request_counts(self, service: str) -> Dict[str, int]:
        """Get request counts for different time windows."""

        now = time.time()
        history = self.request_history[service]

        counts = {
            "minute": 0,
            "hour": 0,
            "day": 0
        }

        # Count requests in each window
        for timestamp in history:
            age_seconds = now - timestamp

            if age_seconds < 60:  # Last minute
                counts["minute"] += 1

            if age_seconds < 3600:  # Last hour
                counts["hour"] += 1

            if age_seconds < 86400:  # Last day
                counts["day"] += 1

        return counts

    def wait_if_needed(self, service: str) -> float:
        """
        Wait if necessary to respect rate limits.

        Args:
            service: Service name

        Returns:
            Wait time in seconds (0 if no wait needed)
        """

        if service not in self.limits:
            logger.warning(f"No rate limit configured for {service}, using default")
            service = "openai"  # Default

        with self.locks[service]:
            self._clean_old_requests(service)

            config = self.limits[service]
            counts = self._get_request_counts(service)
            now = time.time()

            wait_time = 0.0

            # Check minimum interval
            time_since_last = now - self.last_request[service]
            if time_since_last < config.min_interval_seconds:
                wait_time = max(wait_time, config.min_interval_seconds - time_since_last)

            # Check per-minute limit
            if counts["minute"] >= config.requests_per_minute:
                # Wait until oldest request in minute window expires
                oldest_in_minute = next(
                    (ts for ts in self.request_history[service] if now - ts < 60),
                    None
                )

                if oldest_in_minute:
                    wait_until_minute = oldest_in_minute + 60 - now + 0.1  # Small buffer
                    wait_time = max(wait_time, wait_until_minute)

            # Check per-hour limit
            if counts["hour"] >= config.requests_per_hour:
                oldest_in_hour = next(
                    (ts for ts in self.request_history[service] if now - ts < 3600),
                    None
                )

                if oldest_in_hour:
                    wait_until_hour = oldest_in_hour + 3600 - now + 0.1
                    wait_time = max(wait_time, wait_until_hour)

            # Check per-day limit
            if counts["day"] >= config.requests_per_day:
                oldest_in_day = next(
                    (ts for ts in self.request_history[service] if now - ts < 86400),
                    None
                )

                if oldest_in_day:
                    wait_until_day = oldest_in_day + 86400 - now + 0.1
                    wait_time = max(wait_time, wait_until_day)

            # Wait if needed
            if wait_time > 0:
                logger.info(f"⏳ Rate limit: waiting {wait_time:.2f}s for {service}")
                time.sleep(wait_time)

            # Record this request
            self.request_history[service].append(time.time())
            self.last_request[service] = time.time()

            return wait_time

    def execute_with_limit(self, service: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with rate limiting.

        Args:
            service: Service name
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """

        self.wait_if_needed(service)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if it's a rate limit error
            error_str = str(e).lower()

            if "rate limit" in error_str or "too many requests" in error_str:
                logger.warning(f"⚠️ Rate limit hit for {service}, backing off...")

                # Exponential backoff
                backoff_time = 60  # Start with 1 minute
                time.sleep(backoff_time)

                # Retry once
                return func(*args, **kwargs)
            else:
                raise

    def get_status(self, service: str) -> Dict[str, Any]:
        """Get rate limit status for a service."""

        if service not in self.limits:
            return {"error": f"No limits configured for {service}"}

        with self.locks[service]:
            self._clean_old_requests(service)

            config = self.limits[service]
            counts = self._get_request_counts(service)

            return {
                "service": service,
                "limits": {
                    "per_minute": config.requests_per_minute,
                    "per_hour": config.requests_per_hour,
                    "per_day": config.requests_per_day
                },
                "current": counts,
                "remaining": {
                    "minute": max(0, config.requests_per_minute - counts["minute"]),
                    "hour": max(0, config.requests_per_hour - counts["hour"]),
                    "day": max(0, config.requests_per_day - counts["day"])
                },
                "percentage_used": {
                    "minute": round(counts["minute"] / config.requests_per_minute * 100, 1),
                    "hour": round(counts["hour"] / config.requests_per_hour * 100, 1),
                    "day": round(counts["day"] / config.requests_per_day * 100, 1)
                }
            }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all services."""
        return {
            service: self.get_status(service)
            for service in self.limits.keys()
        }


# Global rate limiter instance
_global_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance (singleton)."""
    global _global_limiter

    if _global_limiter is None:
        _global_limiter = RateLimiter()

    return _global_limiter


def main():
    """CLI for rate limiter testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Rate Limiter")
    parser.add_argument("--status", type=str, help="Get status for service")
    parser.add_argument("--all-status", action="store_true", help="Get status for all services")
    parser.add_argument("--test", type=str, help="Test rate limiting for service")
    parser.add_argument("--requests", type=int, default=10, help="Number of test requests")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    limiter = get_rate_limiter()

    if args.status:
        import json
        status = limiter.get_status(args.status)
        print(json.dumps(status, indent=2))

    elif args.all_status:
        import json
        statuses = limiter.get_all_status()
        print(json.dumps(statuses, indent=2))

    elif args.test:
        print(f"\n🧪 Testing rate limiter for {args.test} with {args.requests} requests...\n")

        def dummy_api_call():
            return f"Request at {time.time()}"

        for i in range(args.requests):
            start = time.time()
            result = limiter.execute_with_limit(args.test, dummy_api_call)
            elapsed = time.time() - start

            print(f"Request {i+1}/{args.requests}: {elapsed:.2f}s wait")

        print("\nFinal status:")
        import json
        status = limiter.get_status(args.test)
        print(json.dumps(status, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
