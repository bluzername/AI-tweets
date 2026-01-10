"""
Data models for the OSINT Handle Finder.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json


class LookupStatus(Enum):
    """Status of a handle lookup."""
    VERIFIED = "verified"
    PENDING_REVIEW = "pending_review"
    NOT_FOUND = "not_found"
    MANUAL_APPROVED = "manual_approved"
    MANUAL_REJECTED = "manual_rejected"
    CACHED = "cached"


@dataclass
class HandleLookupContext:
    """
    Context for handle lookup to improve accuracy.

    Providing context helps disambiguate common names and
    verify that found handles match the expected person.
    """
    podcast_name: Optional[str] = None
    episode_title: Optional[str] = None
    episode_description: Optional[str] = None
    known_profession: Optional[str] = None  # e.g., "podcaster", "author", "entrepreneur"
    known_company: Optional[str] = None
    known_location: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HandleLookupContext':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HandleCandidate:
    """
    A potential Twitter handle candidate from a source.

    Each source that finds a potential handle creates one of these
    with supporting evidence.
    """
    handle: str                    # Username (without @)
    source: str                    # Source that found this (e.g., "wikidata", "twitter_api")
    raw_confidence: float          # Source-specific confidence (0-1)
    evidence: Dict[str, Any] = field(default_factory=dict)  # Supporting evidence
    profile_data: Optional[Dict[str, Any]] = None  # Profile info if available

    def __post_init__(self):
        """Normalize handle."""
        # Remove @ if present
        if self.handle.startswith('@'):
            self.handle = self.handle[1:]
        # Lowercase for consistency
        self.handle = self.handle.lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "handle": self.handle,
            "source": self.source,
            "raw_confidence": self.raw_confidence,
            "evidence": self.evidence,
            "profile_data": self.profile_data
        }


@dataclass
class HandleLookupResult:
    """
    Final result of a handle lookup.

    Contains the best handle found (if any), confidence score,
    status, and all candidates considered.
    """
    name: str
    handle: Optional[str] = None   # Final handle or None
    confidence: float = 0.0        # Final confidence score (0-1)
    status: LookupStatus = LookupStatus.NOT_FOUND
    candidates: List[HandleCandidate] = field(default_factory=list)
    review_id: Optional[str] = None  # If sent to review queue
    cached: bool = False           # Was this from cache?
    lookup_time_ms: int = 0        # How long the lookup took
    sources_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "handle": self.handle,
            "confidence": self.confidence,
            "status": self.status.value,
            "candidates": [c.to_dict() for c in self.candidates],
            "review_id": self.review_id,
            "cached": self.cached,
            "lookup_time_ms": self.lookup_time_ms,
            "sources_used": self.sources_used
        }

    @property
    def handle_with_at(self) -> Optional[str]:
        """Get handle with @ prefix."""
        return f"@{self.handle}" if self.handle else None

    @property
    def is_verified(self) -> bool:
        """Check if this result is verified (either auto or manual)."""
        return self.status in [LookupStatus.VERIFIED, LookupStatus.MANUAL_APPROVED]


@dataclass
class ReviewQueueItem:
    """
    An item in the human review queue.

    Created when a lookup finds candidates but confidence is below threshold.
    """
    review_id: str
    name: str
    candidates: List[HandleCandidate]
    context: Optional[HandleLookupContext] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, approved, rejected
    approved_handle: Optional[str] = None
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "review_id": self.review_id,
            "name": self.name,
            "candidates": [c.to_dict() for c in self.candidates],
            "context": self.context.to_dict() if self.context else None,
            "confidence_scores": self.confidence_scores,
            "status": self.status,
            "approved_handle": self.approved_handle,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None
        }


@dataclass
class VerificationResult:
    """
    Result of verifying a handle candidate.
    """
    passed: bool
    overall_score: float  # 0-1 score from verification
    checks: Dict[str, bool] = field(default_factory=dict)  # Individual check results
    reasons: List[str] = field(default_factory=list)  # Reasons for pass/fail
    profile_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "checks": self.checks,
            "reasons": self.reasons,
            "profile_data": self.profile_data
        }


# Type aliases for clarity
HandleCandidateList = List[HandleCandidate]
SourceResults = Dict[str, HandleCandidateList]
