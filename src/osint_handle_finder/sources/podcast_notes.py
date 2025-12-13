"""
Podcast Notes source for extracting Twitter handles.

Parses podcast episode descriptions, show notes, and RSS feed data
to find Twitter handles mentioned in the context of guest appearances.

This is particularly high-confidence when handles are found in official
episode descriptions since podcasters typically verify guest info.
"""

import re
import logging
from typing import List, Optional, Dict, Tuple
from html import unescape

from .base import HandleSource
from ..models import HandleCandidate, HandleLookupContext

logger = logging.getLogger(__name__)


class PodcastNotesSource(HandleSource):
    """
    Extract Twitter handles from podcast episode metadata.

    Parses show notes, descriptions, and episode content for
    Twitter @mentions and profile links.
    """

    priority = 5  # Very high priority - direct context
    base_confidence = 0.80  # High confidence when found in official notes
    requires_auth = False
    source_name = "podcast_notes"

    # Patterns for finding Twitter handles in text
    HANDLE_PATTERNS = [
        # Direct @mentions
        r'@([a-zA-Z0-9_]{1,15})\b',

        # Twitter URLs
        r'(?:twitter\.com|x\.com)/([a-zA-Z0-9_]{1,15})(?:/|$|\?|")',

        # Common formats in show notes
        r'Twitter:\s*@?([a-zA-Z0-9_]{1,15})\b',
        r'X:\s*@?([a-zA-Z0-9_]{1,15})\b',
        r'on Twitter:\s*@?([a-zA-Z0-9_]{1,15})\b',
        r'on X:\s*@?([a-zA-Z0-9_]{1,15})\b',
        r'follow\s+(?:on\s+)?(?:Twitter|X):\s*@?([a-zA-Z0-9_]{1,15})\b',
        r'connect\s+on\s+(?:Twitter|X):\s*@?([a-zA-Z0-9_]{1,15})\b',
    ]

    # Context indicators that increase confidence
    GUEST_INDICATORS = [
        r'guest:?\s*',
        r'featuring:?\s*',
        r'interview(?:ed|ing)?(?:\s+with)?:?\s*',
        r'speaks?\s+with:?\s*',
        r'talks?\s+(?:to|with):?\s*',
        r"today'?s?\s+guest:?\s*",
        r'special\s+guest:?\s*',
        r'joined\s+by:?\s*',
        r'welcom(?:e|ing):?\s*',
    ]

    def is_available(self) -> bool:
        """This source is always available (no external dependencies)."""
        return True

    def lookup(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[HandleCandidate]:
        """
        Extract handles from podcast context.

        Args:
            name: Person name to find handles for
            context: Must contain podcast metadata for this source to work

        Returns:
            List of handle candidates
        """
        if not context:
            logger.debug("No context provided for podcast notes source")
            return []

        candidates = []

        # Combine all text sources
        text_sources = []
        if context.episode_title:
            text_sources.append(("title", context.episode_title))
        if context.episode_description:
            text_sources.append(("description", context.episode_description))

        if not text_sources:
            return []

        # Find handles in each text source
        for source_type, text in text_sources:
            found_handles = self._extract_handles_near_name(text, name)

            for handle, proximity_score, snippet in found_handles:
                confidence = self._calculate_confidence(
                    name, handle, text, source_type, proximity_score
                )

                if confidence > 0.3:
                    evidence = {
                        "source_type": source_type,
                        "proximity_score": proximity_score,
                        "snippet": snippet[:200] if snippet else None,
                        "podcast_name": context.podcast_name,
                        "episode_title": context.episode_title,
                        "match_type": "podcast_notes"
                    }

                    candidates.append(self._create_candidate(
                        handle=handle,
                        confidence=confidence,
                        evidence=evidence
                    ))

        # Deduplicate
        return self._deduplicate_by_handle(candidates)

    def _extract_handles_near_name(
        self,
        text: str,
        name: str
    ) -> List[Tuple[str, float, str]]:
        """
        Extract handles from text, scoring by proximity to the person's name.

        Args:
            text: Text to search
            name: Person name to find near

        Returns:
            List of (handle, proximity_score, context_snippet) tuples
        """
        # Clean and prepare text
        text = unescape(text)
        text_lower = text.lower()
        name_lower = name.lower()

        # Find all occurrences of the name
        name_positions = []
        for match in re.finditer(re.escape(name_lower), text_lower):
            name_positions.append((match.start(), match.end()))

        # Also try partial name matches (first/last name)
        name_parts = name.split()
        if len(name_parts) >= 2:
            for part in [name_parts[0], name_parts[-1]]:
                if len(part) >= 3:  # Skip short parts
                    for match in re.finditer(r'\b' + re.escape(part.lower()) + r'\b', text_lower):
                        name_positions.append((match.start(), match.end()))

        # Find all handles
        found_handles = []
        for pattern in self.HANDLE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                handle = match.group(1).lower()
                handle_pos = match.start()

                if not self._is_valid_handle(handle):
                    continue

                # Calculate proximity to nearest name mention
                proximity_score = 0.0
                if name_positions:
                    min_distance = min(
                        abs(handle_pos - np[0]) for np in name_positions
                    )
                    # Score based on distance (closer = higher score)
                    # Within 100 chars = high score, beyond 500 = low score
                    if min_distance < 50:
                        proximity_score = 1.0
                    elif min_distance < 100:
                        proximity_score = 0.8
                    elif min_distance < 200:
                        proximity_score = 0.6
                    elif min_distance < 500:
                        proximity_score = 0.4
                    else:
                        proximity_score = 0.2
                else:
                    proximity_score = 0.3  # Default if name not found

                # Extract context snippet around the handle
                start = max(0, handle_pos - 50)
                end = min(len(text), handle_pos + 70)
                snippet = text[start:end]

                found_handles.append((handle, proximity_score, snippet))

        return found_handles

    def _calculate_confidence(
        self,
        target_name: str,
        handle: str,
        text: str,
        source_type: str,
        proximity_score: float
    ) -> float:
        """
        Calculate confidence for a handle found in podcast notes.

        Args:
            target_name: Name we're looking for
            handle: Found handle
            text: Full text context
            source_type: "title" or "description"
            proximity_score: Score based on proximity to name

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.4  # Base confidence

        # Proximity to name is very important
        confidence += proximity_score * 0.3

        # Source type matters
        if source_type == "description":
            confidence += 0.1  # Descriptions are more detailed

        # Handle matches name pattern
        target_parts = target_name.lower().split()
        if len(target_parts) >= 2:
            first = target_parts[0]
            last = target_parts[-1]

            if first in handle and last in handle:
                confidence += 0.20
            elif first in handle or last in handle:
                confidence += 0.10

        # Check for guest indicators near the handle
        text_lower = text.lower()
        for indicator in self.GUEST_INDICATORS:
            if re.search(indicator + r'.*@?' + re.escape(handle), text_lower, re.IGNORECASE):
                confidence += 0.15
                break

        # Check if handle appears in a "follow" context
        if re.search(r'follow.*@?' + re.escape(handle), text_lower, re.IGNORECASE):
            confidence += 0.10

        # Check if it's the only handle in the text (more likely to be the guest)
        all_handles = set()
        for pattern in self.HANDLE_PATTERNS[:2]:  # Just @mentions and URLs
            for match in re.finditer(pattern, text, re.IGNORECASE):
                h = match.group(1).lower()
                if self._is_valid_handle(h):
                    all_handles.add(h)

        if len(all_handles) == 1:
            confidence += 0.15
        elif len(all_handles) <= 3:
            confidence += 0.05

        return min(1.0, confidence)

    def _deduplicate_by_handle(
        self,
        candidates: List[HandleCandidate]
    ) -> List[HandleCandidate]:
        """Keep only highest confidence candidate per handle."""
        handle_map = {}
        for c in candidates:
            key = c.handle.lower()
            if key not in handle_map or c.raw_confidence > handle_map[key].raw_confidence:
                handle_map[key] = c
        return list(handle_map.values())

    @staticmethod
    def _is_valid_handle(handle: str) -> bool:
        """Check if handle is valid."""
        if not handle or len(handle) < 2 or len(handle) > 15:
            return False

        if not all(c.isalnum() or c == '_' for c in handle):
            return False

        if handle.isdigit():
            return False

        # Skip common non-person handles
        skip_handles = {
            'twitter', 'x', 'podcast', 'podcasts', 'show',
            'episode', 'spotify', 'apple', 'youtube', 'itunes',
            'subscribe', 'follow', 'link', 'links', 'here'
        }
        if handle.lower() in skip_handles:
            return False

        return True

    def extract_all_handles(self, text: str) -> List[str]:
        """
        Extract all Twitter handles from text (utility method).

        Args:
            text: Text to parse

        Returns:
            List of handles found (deduplicated, lowercase)
        """
        handles = set()
        for pattern in self.HANDLE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                handle = match.group(1).lower()
                if self._is_valid_handle(handle):
                    handles.add(handle)
        return list(handles)
