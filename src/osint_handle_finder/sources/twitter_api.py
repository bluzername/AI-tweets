"""
Twitter API source for finding handles.

Uses the Twitter API v2 Free tier to search for users by name
and verify handle validity.

Rate limits: 15-50 requests per 15-minute window (Free tier)
"""

import os
import requests
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base import HandleSource
from ..models import HandleCandidate, HandleLookupContext

logger = logging.getLogger(__name__)


class TwitterAPISource(HandleSource):
    """
    Twitter API v2 source for handle lookup.

    Uses user search and user lookup endpoints.
    High confidence for exact matches.
    """

    priority = 10  # Highest priority (most authoritative)
    base_confidence = 0.85
    requires_auth = True
    source_name = "twitter_api"

    # API endpoints
    USER_SEARCH_URL = "https://api.twitter.com/2/users/by/username/{username}"
    USER_LOOKUP_URL = "https://api.twitter.com/2/users/by"
    USER_SEARCH_BY_NAME_URL = "https://api.twitter.com/2/users/search"

    # Rate limiting
    RATE_LIMIT_WINDOW = 900  # 15 minutes in seconds
    MAX_REQUESTS_PER_WINDOW = 15  # Conservative for Free tier
    TIMEOUT = 10  # seconds

    def __init__(self):
        """Initialize Twitter API source."""
        super().__init__()
        # Check multiple possible env var names for bearer token
        self.bearer_token = (
            os.getenv("TWITTER_BEARER_TOKEN") or
            os.getenv("MAIN_BEARER_TOKEN") or
            os.getenv("CASUAL_BEARER_TOKEN")
        )
        self._request_times: List[datetime] = []

    def is_available(self) -> bool:
        """Check if Twitter API is available (token present and API reachable)."""
        if not self.bearer_token:
            logger.debug("Twitter bearer token not configured")
            return False

        try:
            # Try a simple API call
            headers = self._get_headers()
            response = requests.get(
                self.USER_SEARCH_URL.format(username="twitter"),
                headers=headers,
                timeout=5
            )
            # 200 = success, 401 = auth error, 429 = rate limited (but API works)
            return response.status_code in [200, 429]
        except Exception as e:
            logger.debug(f"Twitter API check failed: {e}")
            return False

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "PodcastTLDR/1.0",
        }

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            True if we can make a request
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.RATE_LIMIT_WINDOW)

        # Remove old requests
        self._request_times = [t for t in self._request_times if t > cutoff]

        return len(self._request_times) < self.MAX_REQUESTS_PER_WINDOW

    def _record_request(self):
        """Record a request timestamp for rate limiting."""
        self._request_times.append(datetime.utcnow())

    def lookup(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[HandleCandidate]:
        """
        Look up Twitter handles via API.

        Args:
            name: Person name to search for
            context: Optional context for matching

        Returns:
            List of handle candidates
        """
        if not self.bearer_token:
            logger.debug("Twitter API not configured, skipping")
            return []

        if not self._check_rate_limit():
            logger.warning("Twitter API rate limit reached, skipping")
            return []

        candidates = []

        # Strategy 1: Generate likely handle patterns and check if they exist
        patterns = self._generate_handle_patterns(name)

        for pattern in patterns[:5]:  # Limit to avoid rate exhaustion
            if not self._check_rate_limit():
                break

            user_data = self._lookup_username(pattern)
            if user_data:
                confidence = self._calculate_confidence(name, user_data, context)
                if confidence > 0.3:  # Minimum threshold
                    evidence = {
                        "twitter_id": user_data.get("id"),
                        "display_name": user_data.get("name"),
                        "username": user_data.get("username"),
                        "verified": user_data.get("verified", False),
                        "description": user_data.get("description", "")[:200],
                        "public_metrics": user_data.get("public_metrics"),
                        "match_type": "pattern_match"
                    }

                    candidates.append(self._create_candidate(
                        handle=user_data.get("username"),
                        confidence=confidence,
                        evidence=evidence,
                        profile_data=user_data
                    ))

        # Strategy 2: If user search endpoint is available (v2), use it
        # Note: User search requires elevated access, so we try but handle gracefully
        if not candidates and self._check_rate_limit():
            search_results = self._search_users(name)
            for user_data in search_results[:3]:
                confidence = self._calculate_confidence(name, user_data, context)
                if confidence > 0.3:
                    evidence = {
                        "twitter_id": user_data.get("id"),
                        "display_name": user_data.get("name"),
                        "username": user_data.get("username"),
                        "verified": user_data.get("verified", False),
                        "description": user_data.get("description", "")[:200],
                        "match_type": "search_result"
                    }

                    candidates.append(self._create_candidate(
                        handle=user_data.get("username"),
                        confidence=confidence,
                        evidence=evidence,
                        profile_data=user_data
                    ))

        # Deduplicate by handle
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.handle.lower() not in seen:
                seen.add(c.handle.lower())
                unique_candidates.append(c)

        return unique_candidates

    def _lookup_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Look up a specific username via Twitter API.

        Args:
            username: Twitter username to look up

        Returns:
            User data dict or None if not found
        """
        try:
            self._record_request()
            url = self.USER_SEARCH_URL.format(username=username)
            params = {
                "user.fields": "id,name,username,description,verified,public_metrics,created_at,location,url"
            }

            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data")
            elif response.status_code == 404:
                return None
            elif response.status_code == 429:
                logger.warning("Twitter API rate limited")
                return None
            else:
                logger.debug(f"Twitter API returned {response.status_code} for {username}")
                return None

        except requests.Timeout:
            logger.warning("Twitter API timeout")
            return None
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return None

    def _search_users(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for users by name (requires elevated access).

        Args:
            query: Search query (name)

        Returns:
            List of user data dicts
        """
        try:
            self._record_request()

            # Note: User search requires Basic or higher tier
            # This may return 403 on Free tier
            response = requests.get(
                self.USER_SEARCH_BY_NAME_URL,
                headers=self._get_headers(),
                params={
                    "query": query,
                    "max_results": 10,
                    "user.fields": "id,name,username,description,verified,public_metrics"
                },
                timeout=self.TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            elif response.status_code == 403:
                logger.debug("User search not available on Free tier")
                return []
            else:
                return []

        except Exception as e:
            logger.debug(f"User search error: {e}")
            return []

    def _calculate_confidence(
        self,
        target_name: str,
        user_data: Dict[str, Any],
        context: Optional[HandleLookupContext] = None
    ) -> float:
        """
        Calculate confidence that this user matches the target person.

        Args:
            target_name: Name we're looking for
            user_data: Twitter user data
            context: Optional lookup context

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence

        display_name = user_data.get("name", "").lower()
        username = user_data.get("username", "").lower()
        description = user_data.get("description", "").lower()
        target_lower = target_name.lower()

        # Name matching
        if self._name_matches(target_name, display_name, threshold=0.9):
            confidence += 0.3
        elif self._name_matches(target_name, display_name, threshold=0.7):
            confidence += 0.15

        # Username matches name pattern
        target_parts = target_lower.split()
        if len(target_parts) >= 2:
            first = target_parts[0]
            last = target_parts[-1]
            if first in username and last in username:
                confidence += 0.15
            elif first in username or last in username:
                confidence += 0.08

        # Verified accounts get a boost
        if user_data.get("verified"):
            confidence += 0.15

        # Follower count (notable people usually have followers)
        metrics = user_data.get("public_metrics", {})
        followers = metrics.get("followers_count", 0)
        if followers > 10000:
            confidence += 0.10
        elif followers > 1000:
            confidence += 0.05

        # Context matching
        if context:
            # Profession in bio
            if context.known_profession and context.known_profession.lower() in description:
                confidence += 0.15

            # Company in bio
            if context.known_company and context.known_company.lower() in description:
                confidence += 0.10

            # Podcast mention in bio
            if context.podcast_name and context.podcast_name.lower() in description:
                confidence += 0.15

        return min(1.0, confidence)

    def verify_handle(self, handle: str) -> Optional[Dict[str, Any]]:
        """
        Verify a handle exists and get profile data.

        Args:
            handle: Twitter handle to verify

        Returns:
            Profile data if handle exists, None otherwise
        """
        if not self.bearer_token:
            return None

        handle = handle.lstrip('@')
        return self._lookup_username(handle)

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user data by Twitter ID.

        Args:
            user_id: Twitter user ID

        Returns:
            User data or None
        """
        if not self.bearer_token or not self._check_rate_limit():
            return None

        try:
            self._record_request()
            url = f"https://api.twitter.com/2/users/{user_id}"
            params = {
                "user.fields": "id,name,username,description,verified,public_metrics,created_at"
            }

            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self.TIMEOUT
            )

            if response.status_code == 200:
                return response.json().get("data")
            return None

        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
