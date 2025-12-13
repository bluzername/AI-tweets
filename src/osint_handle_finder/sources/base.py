"""
Abstract base class for OSINT handle sources.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from ..models import HandleCandidate, HandleLookupContext

logger = logging.getLogger(__name__)


class HandleSource(ABC):
    """
    Abstract base class for handle lookup sources.

    Each source implementation should:
    1. Set appropriate priority (lower = higher priority)
    2. Set base_confidence based on source reliability
    3. Implement lookup() to return candidates
    4. Implement is_available() to check if source can be used
    """

    # Lower priority = tried first
    priority: int = 50

    # Source-specific base confidence (0-1)
    base_confidence: float = 0.5

    # Whether this source requires API keys or auth
    requires_auth: bool = False

    # Human-readable source name
    source_name: str = "unknown"

    def __init__(self):
        """Initialize the source."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def lookup(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[HandleCandidate]:
        """
        Look up Twitter handles for a person name.

        Args:
            name: Person name to search for
            context: Optional context to improve matching

        Returns:
            List of handle candidates found by this source
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this source is available for use.

        Returns:
            True if the source can be used (API keys present, service reachable, etc.)
        """
        pass

    def get_source_name(self) -> str:
        """Get human-readable source name."""
        return self.source_name or self.__class__.__name__

    def _create_candidate(
        self,
        handle: str,
        confidence: float,
        evidence: dict = None,
        profile_data: dict = None
    ) -> HandleCandidate:
        """
        Helper to create a HandleCandidate.

        Args:
            handle: Twitter handle (with or without @)
            confidence: Confidence score for this candidate
            evidence: Supporting evidence
            profile_data: Profile information if available

        Returns:
            HandleCandidate instance
        """
        return HandleCandidate(
            handle=handle,
            source=self.get_source_name(),
            raw_confidence=min(1.0, confidence * self.base_confidence),
            evidence=evidence or {},
            profile_data=profile_data
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a name for comparison."""
        return name.lower().strip()

    @staticmethod
    def _name_matches(target: str, candidate: str, threshold: float = 0.8) -> bool:
        """
        Check if two names match (fuzzy matching).

        Args:
            target: The name we're looking for
            candidate: The name found from a source
            threshold: Minimum similarity (0-1)

        Returns:
            True if names match above threshold
        """
        target = target.lower().strip()
        candidate = candidate.lower().strip()

        # Exact match
        if target == candidate:
            return True

        # One contains the other
        if target in candidate or candidate in target:
            return True

        # Check if all parts of target are in candidate
        target_parts = target.split()
        candidate_parts = candidate.split()

        if len(target_parts) >= 2 and len(candidate_parts) >= 2:
            # First and last name match
            if (target_parts[0] == candidate_parts[0] and
                target_parts[-1] == candidate_parts[-1]):
                return True

        # Simple similarity check (character overlap)
        target_chars = set(target.replace(' ', ''))
        candidate_chars = set(candidate.replace(' ', ''))

        if len(target_chars) == 0:
            return False

        overlap = len(target_chars & candidate_chars) / len(target_chars)
        return overlap >= threshold

    @staticmethod
    def _generate_handle_patterns(name: str) -> List[str]:
        """
        Generate potential handle patterns from a name.

        Args:
            name: Person name

        Returns:
            List of potential handle patterns (lowercase, no @)
        """
        name = name.lower().strip()
        parts = name.split()
        patterns = []

        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]

            patterns.extend([
                f"{first}{last}",           # elonmusk
                f"{first}_{last}",          # elon_musk
                f"{first}.{last}",          # elon.musk
                f"{first[0]}{last}",        # emusk
                f"{first}{last[0]}",        # elonm
                f"real{first}{last}",       # realelonmusk
                f"the{first}{last}",        # theelonmusk
                f"{first}{last}official",   # elonmuskofficial
                first,                       # elon
                last,                        # musk
            ])

            # Middle name handling
            if len(parts) > 2:
                middle = parts[1]
                patterns.extend([
                    f"{first}{middle[0]}{last}",  # ejohnmusk (if middle name)
                    f"{first}{middle}{last}",     # elonjohnmusk
                ])

        else:
            # Single name
            patterns.append(name.replace(' ', ''))
            patterns.append(name.replace(' ', '_'))

        # Remove duplicates and invalid patterns
        seen = set()
        valid_patterns = []
        for p in patterns:
            p = p.replace(' ', '').replace('.', '')
            if p and p not in seen and len(p) >= 2:
                seen.add(p)
                valid_patterns.append(p)

        return valid_patterns[:15]  # Limit to 15 patterns
