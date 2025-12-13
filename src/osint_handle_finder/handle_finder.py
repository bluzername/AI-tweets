"""
Main OSINT Handle Finder orchestrator.

Coordinates multiple sources, scoring, verification, and caching
to find Twitter handles for podcast guests and hosts.

Usage:
    finder = OSINTHandleFinder()
    result = finder.find_handle("Elon Musk", context=HandleLookupContext(
        podcast_name="Lex Fridman Podcast",
        known_profession="entrepreneur"
    ))
    print(f"Found: @{result.handle} (confidence: {result.confidence:.2%})")
"""

import logging
import time
import uuid
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import (
    HandleCandidate, HandleLookupResult, HandleLookupContext,
    LookupStatus
)
from .database import HandleDatabase
from .confidence import ConfidenceScorer
from .verification import HandleVerifier
from .sources import (
    ALL_SOURCES, HandleSource,
    WikidataSource, TwitterAPISource, DuckDuckGoSource,
    PodcastNotesSource, LLMInferenceSource
)

logger = logging.getLogger(__name__)


class OSINTHandleFinder:
    """
    Main orchestrator for OSINT-based Twitter handle discovery.

    Features:
    - Multi-source lookups with priority ordering
    - Confidence scoring with source weights
    - Verification layer
    - Caching with positive/negative cache
    - Human review queue for uncertain matches
    """

    # Confidence thresholds (lowered for better recall)
    AUTO_APPROVE_THRESHOLD = 0.60  # Accept handles at 60%+ confidence
    REVIEW_THRESHOLD = 0.45        # Queue 45-60% for review
    MIN_CONFIDENCE = 0.35          # Consider candidates at 35%+

    # Performance settings
    MAX_PARALLEL_SOURCES = 3
    SOURCE_TIMEOUT = 15  # seconds (reduced for faster pipeline)

    def __init__(
        self,
        db_path: str = "data/handles.db",
        auto_approve: bool = True,
        use_review_queue: bool = True,
        verify_handles: bool = True,
        parallel_lookups: bool = True
    ):
        """
        Initialize the handle finder.

        Args:
            db_path: Path to SQLite database
            auto_approve: Automatically approve high-confidence results
            use_review_queue: Queue uncertain results for human review
            verify_handles: Run verification checks on candidates
            parallel_lookups: Run source lookups in parallel
        """
        self.db = HandleDatabase(db_path)
        self.scorer = ConfidenceScorer()
        self.verifier = HandleVerifier()

        self.auto_approve = auto_approve
        self.use_review_queue = use_review_queue
        self.verify_handles = verify_handles
        self.parallel_lookups = parallel_lookups

        # Initialize sources
        self.sources: List[HandleSource] = []
        self._init_sources()

        logger.info(f"Initialized OSINT Handle Finder with {len(self.sources)} sources")

    def _init_sources(self):
        """Initialize all available sources."""
        for source_cls in ALL_SOURCES:
            try:
                source = source_cls()
                if source.is_available() or not source.requires_auth:
                    self.sources.append(source)
                    logger.debug(f"Enabled source: {source.source_name}")
                else:
                    logger.debug(f"Source unavailable (requires auth): {source.source_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize source {source_cls.__name__}: {e}")

        # Sort by priority
        self.sources.sort(key=lambda s: s.priority)

    def find_handle(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None,
        force_refresh: bool = False,
        episode_id: Optional[str] = None
    ) -> HandleLookupResult:
        """
        Find a Twitter handle for a person.

        Args:
            name: Person name to look up
            context: Optional context to improve accuracy
            force_refresh: Skip cache and do fresh lookup
            episode_id: Optional episode ID for logging

        Returns:
            HandleLookupResult with best handle and confidence
        """
        start_time = time.time()
        logger.info(f"Looking up handle for: {name}")

        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = self.db.get_cached_handle(name)
            if cached:
                logger.debug(f"Cache hit for {name}: @{cached.handle}")
                cached.lookup_time_ms = int((time.time() - start_time) * 1000)
                self.db.log_lookup(name, cached, episode_id)
                return cached

        # Gather candidates from all sources
        all_candidates = self._gather_candidates(name, context)

        if not all_candidates:
            logger.info(f"No candidates found for {name}")
            result = HandleLookupResult(
                name=name,
                status=LookupStatus.NOT_FOUND,
                lookup_time_ms=int((time.time() - start_time) * 1000),
                sources_used=[s.source_name for s in self.sources]
            )
            self.db.cache_negative(name, "no_candidates")
            self.db.log_lookup(name, result, episode_id)
            return result

        # Score and rank candidates
        ranked = self.scorer.rank_candidates(all_candidates, name, context)
        logger.debug(f"Ranked {len(ranked)} unique handles for {name}")

        # Get best candidate
        best_handle, best_score, best_sources = ranked[0] if ranked else (None, 0.0, [])

        # Verify top candidate if enabled
        verification_passed = True
        if self.verify_handles and best_handle:
            # Get profile data from candidates
            profile_data = None
            for c in all_candidates:
                if c.handle.lower() == best_handle.lower() and c.profile_data:
                    profile_data = c.profile_data
                    break

            verification = self.verifier.verify(
                handle=best_handle,
                target_name=name,
                context=context,
                profile_data=profile_data
            )

            if not verification.passed:
                logger.warning(f"Verification failed for @{best_handle}: {verification.reasons}")
                verification_passed = False
                # Adjust score downward
                best_score *= 0.7

        # Determine result status
        status = self._determine_status(best_score, verification_passed)

        # Create result
        result = HandleLookupResult(
            name=name,
            handle=best_handle if best_score >= self.MIN_CONFIDENCE else None,
            confidence=best_score,
            status=status,
            candidates=all_candidates,
            lookup_time_ms=int((time.time() - start_time) * 1000),
            sources_used=list(set(c.source for c in all_candidates))
        )

        # Handle based on status
        if status == LookupStatus.VERIFIED:
            # Auto-approve high confidence
            logger.info(f"Auto-approved: @{best_handle} for {name} (conf: {best_score:.2%})")
            self.db.cache_handle(name, result)

        elif status == LookupStatus.PENDING_REVIEW:
            if self.use_review_queue:
                # Add to review queue
                review_id = f"review_{uuid.uuid4().hex[:8]}"
                result.review_id = review_id

                confidence_scores = {h: s for h, s, _ in ranked[:5]}
                self.db.add_to_review_queue(
                    review_id=review_id,
                    name=name,
                    candidates=all_candidates,
                    context=context,
                    confidence_scores=confidence_scores,
                    priority=int(best_score * 100)  # Higher score = higher priority
                )
                logger.info(f"Added to review queue: {name} (review_id: {review_id})")

        else:
            # Not found - cache negative
            self.db.cache_negative(name, "low_confidence")

        # Log the lookup
        self.db.log_lookup(name, result, episode_id)

        return result

    def _gather_candidates(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[HandleCandidate]:
        """
        Gather candidates from all sources.

        Args:
            name: Person name
            context: Optional context

        Returns:
            List of all candidates from all sources
        """
        all_candidates = []

        if self.parallel_lookups and len(self.sources) > 1:
            # Parallel lookups
            with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_SOURCES) as executor:
                futures = {
                    executor.submit(self._safe_lookup, source, name, context): source
                    for source in self.sources
                }

                for future in as_completed(futures, timeout=self.SOURCE_TIMEOUT):
                    source = futures[future]
                    try:
                        candidates = future.result()
                        if candidates:
                            all_candidates.extend(candidates)
                            logger.debug(f"{source.source_name}: found {len(candidates)} candidates")
                    except Exception as e:
                        logger.error(f"Source {source.source_name} failed: {e}")
        else:
            # Sequential lookups
            for source in self.sources:
                candidates = self._safe_lookup(source, name, context)
                if candidates:
                    all_candidates.extend(candidates)
                    logger.debug(f"{source.source_name}: found {len(candidates)} candidates")

        return all_candidates

    def _safe_lookup(
        self,
        source: HandleSource,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[HandleCandidate]:
        """
        Safely perform a source lookup with error handling.

        Args:
            source: Source to query
            name: Person name
            context: Optional context

        Returns:
            List of candidates (empty on error)
        """
        try:
            return source.lookup(name, context)
        except Exception as e:
            logger.error(f"Error in {source.source_name} lookup: {e}")
            return []

    def _determine_status(self, score: float, verification_passed: bool) -> LookupStatus:
        """
        Determine the lookup status based on score and verification.

        Args:
            score: Confidence score
            verification_passed: Whether verification checks passed

        Returns:
            LookupStatus enum value
        """
        if score >= self.AUTO_APPROVE_THRESHOLD and verification_passed:
            return LookupStatus.VERIFIED

        if score >= self.REVIEW_THRESHOLD:
            return LookupStatus.PENDING_REVIEW

        if score >= self.MIN_CONFIDENCE:
            return LookupStatus.PENDING_REVIEW

        return LookupStatus.NOT_FOUND

    def find_handles_batch(
        self,
        names: List[str],
        context: Optional[HandleLookupContext] = None,
        episode_id: Optional[str] = None
    ) -> Dict[str, HandleLookupResult]:
        """
        Find handles for multiple names.

        Args:
            names: List of names to look up
            context: Shared context for all lookups
            episode_id: Optional episode ID

        Returns:
            Dict mapping name -> result
        """
        results = {}
        for name in names:
            results[name] = self.find_handle(name, context, episode_id=episode_id)
        return results

    def approve_review(
        self,
        review_id: str,
        approved_handle: str,
        reviewer: str = "system",
        notes: str = ""
    ) -> bool:
        """
        Approve a review queue item.

        Args:
            review_id: Review ID
            approved_handle: The handle to approve
            reviewer: Who approved it
            notes: Optional notes

        Returns:
            True if successful
        """
        return self.db.approve_review(review_id, approved_handle, reviewer, notes)

    def reject_review(
        self,
        review_id: str,
        reviewer: str = "system",
        notes: str = ""
    ) -> bool:
        """
        Reject a review queue item.

        Args:
            review_id: Review ID
            reviewer: Who rejected it
            notes: Reason for rejection

        Returns:
            True if successful
        """
        return self.db.reject_review(review_id, reviewer, notes)

    def get_pending_reviews(self, limit: int = 50) -> List:
        """Get pending review items."""
        return self.db.get_pending_reviews(limit)

    def get_stats(self) -> Dict[str, Any]:
        """Get finder statistics."""
        db_stats = self.db.get_stats()
        return {
            **db_stats,
            "active_sources": [s.source_name for s in self.sources],
            "source_count": len(self.sources),
            "auto_approve_threshold": self.AUTO_APPROVE_THRESHOLD,
            "review_threshold": self.REVIEW_THRESHOLD
        }

    def invalidate_cache(self, name: str):
        """Remove a name from cache."""
        self.db.invalidate_cache(name)

    def cleanup(self):
        """Clean up expired cache entries."""
        return self.db.cleanup_expired()


# Convenience function for simple lookups
def find_twitter_handle(
    name: str,
    podcast_name: Optional[str] = None,
    profession: Optional[str] = None,
    company: Optional[str] = None
) -> Optional[str]:
    """
    Simple function to find a Twitter handle.

    Args:
        name: Person name
        podcast_name: Optional podcast name for context
        profession: Optional profession
        company: Optional company

    Returns:
        Twitter handle (without @) or None
    """
    finder = OSINTHandleFinder()

    context = None
    if podcast_name or profession or company:
        context = HandleLookupContext(
            podcast_name=podcast_name,
            known_profession=profession,
            known_company=company
        )

    result = finder.find_handle(name, context)

    if result.status in [LookupStatus.VERIFIED, LookupStatus.PENDING_REVIEW]:
        return result.handle

    return None
