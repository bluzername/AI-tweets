"""
Confidence scoring engine for handle candidates.

Combines evidence from multiple sources to produce a final confidence
score for each handle candidate. Uses source weights, agreement bonuses,
and context-based adjustments.

Formula: final_score = Σ(source_weight × raw_confidence) × agreement_boost × name_match_factor
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .models import HandleCandidate, HandleLookupContext

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Calculates final confidence scores for handle candidates.

    Combines evidence from multiple sources with weighted scoring
    and agreement bonuses.
    """

    # Source weights (sum to ~1.0 for normalization)
    SOURCE_WEIGHTS = {
        "wikidata": 0.30,        # Very reliable for notable people
        "twitter_api": 0.25,     # Direct validation
        "podcast_notes": 0.20,   # High-quality contextual data
        "duckduckgo": 0.15,      # Web search - noisier
        "llm_inference": 0.10,   # AI guess - lowest weight
    }

    # Default weight for unknown sources
    DEFAULT_WEIGHT = 0.10

    # Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    REVIEW_THRESHOLD = 0.70
    LOW_CONFIDENCE_THRESHOLD = 0.50

    # Agreement bonus: multiple sources finding the same handle
    AGREEMENT_BONUS = {
        2: 1.15,  # 2 sources agree: 15% bonus
        3: 1.25,  # 3 sources agree: 25% bonus
        4: 1.35,  # 4 sources agree: 35% bonus
        5: 1.40,  # 5+ sources agree: 40% bonus
    }

    def __init__(self, verification_boost: float = 1.2):
        """
        Initialize the scorer.

        Args:
            verification_boost: Multiplier for verified handles
        """
        self.verification_boost = verification_boost

    def score_candidates(
        self,
        candidates: List[HandleCandidate],
        target_name: str,
        context: Optional[HandleLookupContext] = None
    ) -> Dict[str, float]:
        """
        Calculate final confidence scores for all candidates.

        Args:
            candidates: List of candidates from all sources
            target_name: The person name we're looking for
            context: Optional lookup context

        Returns:
            Dict mapping handle -> final_confidence_score
        """
        if not candidates:
            return {}

        # Group candidates by handle
        handle_groups = self._group_by_handle(candidates)

        # Calculate scores for each unique handle
        scores = {}
        for handle, group_candidates in handle_groups.items():
            scores[handle] = self._calculate_final_score(
                handle, group_candidates, target_name, context
            )

        return scores

    def rank_candidates(
        self,
        candidates: List[HandleCandidate],
        target_name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Rank candidates by confidence score.

        Args:
            candidates: List of candidates
            target_name: Person name
            context: Optional context

        Returns:
            List of (handle, score, sources) tuples, sorted by score desc
        """
        scores = self.score_candidates(candidates, target_name, context)

        # Get sources for each handle
        handle_sources = defaultdict(set)
        for c in candidates:
            handle_sources[c.handle.lower()].add(c.source)

        # Create ranked list
        ranked = [
            (handle, score, list(handle_sources[handle]))
            for handle, score in scores.items()
        ]

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def get_best_candidate(
        self,
        candidates: List[HandleCandidate],
        target_name: str,
        context: Optional[HandleLookupContext] = None,
        min_confidence: float = 0.50
    ) -> Optional[Tuple[str, float, List[str]]]:
        """
        Get the best candidate above minimum confidence.

        Args:
            candidates: List of candidates
            target_name: Person name
            context: Optional context
            min_confidence: Minimum required confidence

        Returns:
            (handle, score, sources) tuple or None
        """
        ranked = self.rank_candidates(candidates, target_name, context)

        if ranked and ranked[0][1] >= min_confidence:
            return ranked[0]

        return None

    def _group_by_handle(
        self,
        candidates: List[HandleCandidate]
    ) -> Dict[str, List[HandleCandidate]]:
        """Group candidates by normalized handle."""
        groups = defaultdict(list)
        for c in candidates:
            groups[c.handle.lower()].append(c)
        return dict(groups)

    def _calculate_final_score(
        self,
        handle: str,
        candidates: List[HandleCandidate],
        target_name: str,
        context: Optional[HandleLookupContext] = None
    ) -> float:
        """
        Calculate final confidence score for a handle.

        Args:
            handle: The handle
            candidates: All candidates for this handle
            target_name: Person name
            context: Optional context

        Returns:
            Final confidence score (0-1)
        """
        # Calculate weighted sum of source confidences
        weighted_sum = 0.0
        total_weight = 0.0

        for c in candidates:
            weight = self.SOURCE_WEIGHTS.get(c.source, self.DEFAULT_WEIGHT)
            weighted_sum += weight * c.raw_confidence
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalize
        base_score = weighted_sum / total_weight

        # Apply agreement bonus
        num_sources = len(set(c.source for c in candidates))
        agreement_multiplier = self.AGREEMENT_BONUS.get(
            min(num_sources, 5), 1.0
        )
        score = base_score * agreement_multiplier

        # Apply name match factor
        name_factor = self._calculate_name_match_factor(handle, target_name)
        score *= name_factor

        # Apply context boost if available
        if context:
            context_boost = self._calculate_context_boost(candidates, context)
            score *= context_boost

        # Check for verification evidence
        if self._has_verification_evidence(candidates):
            score *= self.verification_boost

        # Cap at 1.0
        return min(1.0, score)

    def _calculate_name_match_factor(
        self,
        handle: str,
        target_name: str
    ) -> float:
        """
        Calculate how well the handle matches the target name.

        Args:
            handle: Twitter handle
            target_name: Person name

        Returns:
            Match factor (0.8 - 1.2)
        """
        handle_lower = handle.lower()
        name_lower = target_name.lower()
        name_parts = name_lower.split()

        # Strong match: both first and last name in handle
        if len(name_parts) >= 2:
            first = name_parts[0]
            last = name_parts[-1]

            # Both names present
            if first in handle_lower and last in handle_lower:
                return 1.15

            # First initial + last name (e.g., jsmith for John Smith)
            if first[0] + last == handle_lower or first + last[0] == handle_lower:
                return 1.10

            # One name present
            if first in handle_lower or last in handle_lower:
                return 1.05

        # Single name case
        else:
            name_clean = name_lower.replace(' ', '')
            if name_clean in handle_lower or handle_lower in name_clean:
                return 1.10

        # Handle looks unrelated to name
        # Check for any character overlap
        name_chars = set(name_lower.replace(' ', ''))
        handle_chars = set(handle_lower)

        overlap = len(name_chars & handle_chars) / max(len(name_chars), 1)

        if overlap < 0.3:
            return 0.85  # Penalize unrelated handles
        elif overlap < 0.5:
            return 0.95

        return 1.0

    def _calculate_context_boost(
        self,
        candidates: List[HandleCandidate],
        context: HandleLookupContext
    ) -> float:
        """
        Calculate boost based on context matches in evidence.

        Args:
            candidates: Candidates for this handle
            context: Lookup context

        Returns:
            Context boost multiplier (1.0 - 1.2)
        """
        boost = 1.0

        for c in candidates:
            evidence = c.evidence or {}
            profile = c.profile_data or {}

            # Check profile bio against context
            bio = profile.get('description', '').lower()

            if context.podcast_name:
                if context.podcast_name.lower() in bio:
                    boost = max(boost, 1.15)

            if context.known_profession:
                if context.known_profession.lower() in bio:
                    boost = max(boost, 1.10)

            if context.known_company:
                if context.known_company.lower() in bio:
                    boost = max(boost, 1.10)

            # Check evidence from podcast notes
            if c.source == 'podcast_notes':
                proximity = evidence.get('proximity_score', 0)
                if proximity > 0.8:
                    boost = max(boost, 1.15)

        return boost

    def _has_verification_evidence(
        self,
        candidates: List[HandleCandidate]
    ) -> bool:
        """
        Check if any candidate has verification evidence.

        Args:
            candidates: Candidates to check

        Returns:
            True if verified
        """
        for c in candidates:
            evidence = c.evidence or {}
            profile = c.profile_data or {}

            # Twitter verified badge
            if profile.get('verified'):
                return True

            # Wikidata link (official)
            if c.source == 'wikidata' and evidence.get('wikidata_id'):
                return True

        return False

    def needs_review(self, score: float) -> bool:
        """
        Check if a score requires human review.

        Args:
            score: Confidence score

        Returns:
            True if needs review
        """
        return self.LOW_CONFIDENCE_THRESHOLD <= score < self.HIGH_CONFIDENCE_THRESHOLD

    def is_high_confidence(self, score: float) -> bool:
        """Check if score is high confidence (auto-approve)."""
        return score >= self.HIGH_CONFIDENCE_THRESHOLD

    def is_low_confidence(self, score: float) -> bool:
        """Check if score is too low to use."""
        return score < self.LOW_CONFIDENCE_THRESHOLD

    def explain_score(
        self,
        handle: str,
        candidates: List[HandleCandidate],
        target_name: str,
        context: Optional[HandleLookupContext] = None
    ) -> Dict:
        """
        Get detailed explanation of how a score was calculated.

        Args:
            handle: The handle
            candidates: Candidates for this handle
            target_name: Person name
            context: Optional context

        Returns:
            Explanation dict
        """
        handle_candidates = [c for c in candidates if c.handle.lower() == handle.lower()]

        if not handle_candidates:
            return {"error": "No candidates found for handle"}

        # Calculate component scores
        source_contributions = {}
        for c in handle_candidates:
            weight = self.SOURCE_WEIGHTS.get(c.source, self.DEFAULT_WEIGHT)
            source_contributions[c.source] = {
                "raw_confidence": c.raw_confidence,
                "weight": weight,
                "weighted_score": weight * c.raw_confidence
            }

        num_sources = len(source_contributions)
        agreement_multiplier = self.AGREEMENT_BONUS.get(min(num_sources, 5), 1.0)
        name_factor = self._calculate_name_match_factor(handle, target_name)

        context_boost = 1.0
        if context:
            context_boost = self._calculate_context_boost(handle_candidates, context)

        verification_boost = self.verification_boost if self._has_verification_evidence(handle_candidates) else 1.0

        final_score = self._calculate_final_score(handle, handle_candidates, target_name, context)

        return {
            "handle": handle,
            "final_score": final_score,
            "components": {
                "source_contributions": source_contributions,
                "num_sources": num_sources,
                "agreement_multiplier": agreement_multiplier,
                "name_match_factor": name_factor,
                "context_boost": context_boost,
                "verification_boost": verification_boost
            },
            "threshold_status": {
                "high_confidence": self.is_high_confidence(final_score),
                "needs_review": self.needs_review(final_score),
                "low_confidence": self.is_low_confidence(final_score)
            }
        }
