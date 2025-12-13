"""
Verification layer for Twitter handle candidates.

Performs additional checks to verify that a handle candidate
is valid and likely belongs to the target person:
- Account exists and is active
- Display name matches target
- Not a parody/fan account
- Bio matches known context
"""

import os
import re
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from .models import HandleCandidate, HandleLookupContext, VerificationResult

logger = logging.getLogger(__name__)


class HandleVerifier:
    """
    Verifies handle candidates through multiple checks.

    Uses Twitter API (when available) and heuristic analysis
    to verify handle validity.
    """

    # Parody account indicators
    PARODY_INDICATORS = [
        'parody', 'fan account', 'fan page', 'not affiliated',
        'unofficial', 'satire', 'roleplay', 'rp', 'fake',
        'tribute', 'updates', 'news about', 'facts about'
    ]

    # Minimum account age (days) to consider legitimate
    MIN_ACCOUNT_AGE_DAYS = 30

    # Minimum followers for notable people
    MIN_FOLLOWERS_NOTABLE = 100

    def __init__(self, twitter_source=None):
        """
        Initialize verifier.

        Args:
            twitter_source: Optional TwitterAPISource instance for API calls
        """
        self.twitter_source = twitter_source
        self._cache: Dict[str, Dict] = {}

    def verify(
        self,
        handle: str,
        target_name: str,
        context: Optional[HandleLookupContext] = None,
        profile_data: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify a handle candidate.

        Args:
            handle: Twitter handle to verify
            target_name: Person name we're looking for
            context: Optional lookup context
            profile_data: Optional pre-fetched profile data

        Returns:
            VerificationResult with pass/fail and details
        """
        handle = handle.lstrip('@').lower()

        checks = {}
        reasons = []
        total_score = 0.0
        max_score = 0.0

        # Get profile data if not provided
        if not profile_data:
            profile_data = self._get_profile(handle)

        if not profile_data:
            return VerificationResult(
                passed=False,
                overall_score=0.0,
                checks={"profile_exists": False},
                reasons=["Could not fetch profile data"],
                profile_data=None
            )

        checks["profile_exists"] = True
        total_score += 1.0
        max_score += 1.0

        # Check 1: Account is active (not suspended/deactivated)
        max_score += 1.0
        if self._is_account_active(profile_data):
            checks["account_active"] = True
            total_score += 1.0
        else:
            checks["account_active"] = False
            reasons.append("Account appears inactive or suspended")

        # Check 2: Display name matches target
        max_score += 1.5
        name_match_score = self._check_name_match(
            target_name, profile_data.get('name', '')
        )
        checks["name_match"] = name_match_score > 0.5
        total_score += name_match_score * 1.5
        if name_match_score < 0.5:
            reasons.append(f"Display name '{profile_data.get('name', '')}' doesn't match target '{target_name}'")

        # Check 3: Not a parody/fan account
        max_score += 1.0
        is_parody = self._check_parody_indicators(profile_data)
        checks["not_parody"] = not is_parody
        if is_parody:
            reasons.append("Bio contains parody/fan account indicators")
        else:
            total_score += 1.0

        # Check 4: Bio matches context (if context provided)
        if context:
            max_score += 1.0
            context_match = self._check_context_match(profile_data, context)
            checks["context_match"] = context_match > 0.5
            total_score += context_match
            if context_match < 0.5:
                reasons.append("Bio doesn't match expected context")

        # Check 5: Account age (newer accounts more suspicious)
        max_score += 0.5
        age_score = self._check_account_age(profile_data)
        checks["account_age_ok"] = age_score > 0.5
        total_score += age_score * 0.5
        if age_score < 0.5:
            reasons.append("Account is very new")

        # Check 6: Follower count (notable people usually have followers)
        max_score += 0.5
        follower_score = self._check_follower_count(profile_data)
        checks["follower_count_ok"] = follower_score > 0.3
        total_score += follower_score * 0.5
        if follower_score < 0.3:
            reasons.append("Very low follower count for a notable person")

        # Calculate overall score
        overall_score = total_score / max_score if max_score > 0 else 0.0

        # Determine if passed
        # Must have: profile exists, account active, reasonable name match, not parody
        critical_passed = (
            checks.get("profile_exists", False) and
            checks.get("account_active", False) and
            checks.get("name_match", False) and
            checks.get("not_parody", False)
        )

        passed = critical_passed and overall_score >= 0.6

        return VerificationResult(
            passed=passed,
            overall_score=overall_score,
            checks=checks,
            reasons=reasons if not passed else ["All verification checks passed"],
            profile_data=profile_data
        )

    def verify_batch(
        self,
        candidates: List[HandleCandidate],
        target_name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[tuple]:
        """
        Verify multiple candidates.

        Args:
            candidates: List of candidates to verify
            target_name: Target person name
            context: Optional context

        Returns:
            List of (candidate, verification_result) tuples
        """
        results = []
        for candidate in candidates:
            result = self.verify(
                handle=candidate.handle,
                target_name=target_name,
                context=context,
                profile_data=candidate.profile_data
            )
            results.append((candidate, result))
        return results

    def _get_profile(self, handle: str) -> Optional[Dict[str, Any]]:
        """
        Get profile data for a handle.

        Args:
            handle: Twitter handle

        Returns:
            Profile data dict or None
        """
        # Check cache
        if handle in self._cache:
            return self._cache[handle]

        # Try Twitter API if available
        if self.twitter_source and self.twitter_source.is_available():
            profile = self.twitter_source.verify_handle(handle)
            if profile:
                self._cache[handle] = profile
                return profile

        return None

    def _is_account_active(self, profile: Dict[str, Any]) -> bool:
        """Check if account appears active."""
        # If we got profile data, account exists
        if profile:
            # Check if suspended
            if profile.get('suspended', False):
                return False

            # Check if protected and has any activity
            metrics = profile.get('public_metrics', {})
            if metrics.get('tweet_count', 0) > 0:
                return True

            return True

        return False

    def _check_name_match(self, target: str, display_name: str) -> float:
        """
        Check how well display name matches target.

        Args:
            target: Target name
            display_name: Profile display name

        Returns:
            Match score (0-1)
        """
        if not display_name:
            return 0.0

        target_lower = target.lower().strip()
        display_lower = display_name.lower().strip()

        # Exact match
        if target_lower == display_lower:
            return 1.0

        # One contains the other
        if target_lower in display_lower or display_lower in target_lower:
            return 0.85

        # Check name parts
        target_parts = target_lower.split()
        display_parts = display_lower.split()

        if len(target_parts) >= 2 and len(display_parts) >= 2:
            # First and last name match
            if target_parts[0] == display_parts[0] and target_parts[-1] == display_parts[-1]:
                return 0.95

            # At least first or last name matches
            if target_parts[0] in display_parts or target_parts[-1] in display_parts:
                return 0.7

        # Single name match
        elif len(target_parts) == 1:
            if target_parts[0] in display_parts:
                return 0.8

        # Character overlap check
        target_chars = set(target_lower.replace(' ', ''))
        display_chars = set(display_lower.replace(' ', ''))

        if len(target_chars) > 0:
            overlap = len(target_chars & display_chars) / len(target_chars)
            if overlap > 0.7:
                return 0.6

        return 0.2

    def _check_parody_indicators(self, profile: Dict[str, Any]) -> bool:
        """
        Check if profile appears to be parody/fan account.

        Args:
            profile: Profile data

        Returns:
            True if parody indicators found
        """
        bio = profile.get('description', '').lower()
        name = profile.get('name', '').lower()

        text_to_check = f"{bio} {name}"

        for indicator in self.PARODY_INDICATORS:
            if indicator in text_to_check:
                return True

        return False

    def _check_context_match(
        self,
        profile: Dict[str, Any],
        context: HandleLookupContext
    ) -> float:
        """
        Check if profile bio matches expected context.

        Args:
            profile: Profile data
            context: Expected context

        Returns:
            Match score (0-1)
        """
        bio = profile.get('description', '').lower()

        if not bio:
            return 0.5  # Neutral if no bio

        score = 0.5  # Start neutral
        matches = 0
        total_checks = 0

        if context.known_profession:
            total_checks += 1
            if context.known_profession.lower() in bio:
                matches += 1
                score += 0.15

        if context.known_company:
            total_checks += 1
            if context.known_company.lower() in bio:
                matches += 1
                score += 0.15

        if context.podcast_name:
            total_checks += 1
            if context.podcast_name.lower() in bio:
                matches += 1
                score += 0.20

        # Professional keywords
        professional_keywords = [
            'author', 'founder', 'ceo', 'host', 'speaker',
            'entrepreneur', 'investor', 'professor', 'writer',
            'journalist', 'producer', 'director', 'creator'
        ]

        for keyword in professional_keywords:
            if keyword in bio:
                score += 0.05
                break

        return min(1.0, score)

    def _check_account_age(self, profile: Dict[str, Any]) -> float:
        """
        Check account age.

        Args:
            profile: Profile data

        Returns:
            Score based on account age (0-1)
        """
        created_at = profile.get('created_at')

        if not created_at:
            return 0.5  # Neutral if unknown

        try:
            # Parse Twitter timestamp format
            if isinstance(created_at, str):
                # Try ISO format first
                try:
                    created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except ValueError:
                    # Try Twitter format
                    created = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')

                age_days = (datetime.now(created.tzinfo) - created).days

                if age_days > 365 * 5:  # 5+ years
                    return 1.0
                elif age_days > 365 * 2:  # 2-5 years
                    return 0.9
                elif age_days > 365:  # 1-2 years
                    return 0.8
                elif age_days > self.MIN_ACCOUNT_AGE_DAYS:
                    return 0.6
                else:
                    return 0.3

        except Exception as e:
            logger.debug(f"Could not parse account age: {e}")
            return 0.5

        return 0.5

    def _check_follower_count(self, profile: Dict[str, Any]) -> float:
        """
        Check follower count for notable people.

        Args:
            profile: Profile data

        Returns:
            Score based on follower count (0-1)
        """
        metrics = profile.get('public_metrics', {})
        followers = metrics.get('followers_count', 0)

        if followers > 100000:
            return 1.0
        elif followers > 10000:
            return 0.9
        elif followers > 1000:
            return 0.7
        elif followers > self.MIN_FOLLOWERS_NOTABLE:
            return 0.5
        elif followers > 10:
            return 0.3
        else:
            return 0.2

    def quick_verify(self, handle: str) -> bool:
        """
        Quick check if a handle exists and is active.

        Args:
            handle: Twitter handle

        Returns:
            True if handle appears valid
        """
        profile = self._get_profile(handle.lstrip('@').lower())
        return profile is not None and self._is_account_active(profile)
