"""
Database layer for the OSINT Handle Finder.
Handles caching, review queue storage, and lookup history.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from .models import (
    HandleCandidate, HandleLookupResult, HandleLookupContext,
    ReviewQueueItem, LookupStatus
)

logger = logging.getLogger(__name__)


class HandleDatabase:
    """
    SQLite database for handle caching and review queue.
    """

    def __init__(self, db_path: str = "data/handles.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Handle cache table - stores verified handles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS handle_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    name_normalized TEXT NOT NULL UNIQUE,
                    handle TEXT,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    sources_json TEXT,
                    evidence_json TEXT,
                    verified_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)

            # Review queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    name_normalized TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    context_json TEXT,
                    confidence_scores_json TEXT,
                    priority INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    approved_handle TEXT,
                    reviewer TEXT,
                    review_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TIMESTAMP
                )
            """)

            # Negative cache - names where no handle was found
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS negative_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    name_normalized TEXT NOT NULL UNIQUE,
                    lookup_count INTEGER DEFAULT 1,
                    last_lookup TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT,
                    expires_at TIMESTAMP
                )
            """)

            # Lookup history for analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lookup_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    handle TEXT,
                    confidence REAL,
                    status TEXT,
                    sources_used TEXT,
                    duration_ms INTEGER,
                    cached BOOLEAN DEFAULT FALSE,
                    episode_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_normalized
                ON handle_cache(name_normalized)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires
                ON handle_cache(expires_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_review_status
                ON review_queue(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_negative_normalized
                ON negative_cache(name_normalized)
            """)

            logger.info(f"Handle database initialized: {self.db_path}")

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a name for consistent lookups."""
        return name.lower().strip()

    # =========================================================================
    # Cache Operations
    # =========================================================================

    def get_cached_handle(self, name: str) -> Optional[HandleLookupResult]:
        """
        Get a cached handle lookup result.

        Args:
            name: Person name to look up

        Returns:
            Cached result or None if not found/expired
        """
        name_normalized = self._normalize_name(name)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check positive cache
            cursor.execute("""
                SELECT handle, confidence, status, sources_json, evidence_json
                FROM handle_cache
                WHERE name_normalized = ?
                AND (expires_at IS NULL OR expires_at > ?)
            """, (name_normalized, datetime.utcnow().isoformat()))

            row = cursor.fetchone()
            if row:
                candidates = []
                if row['sources_json']:
                    try:
                        sources_data = json.loads(row['sources_json'])
                        for src in sources_data:
                            candidates.append(HandleCandidate(
                                handle=src.get('handle', row['handle']),
                                source=src.get('source', 'cache'),
                                raw_confidence=src.get('confidence', row['confidence']),
                                evidence=src.get('evidence', {})
                            ))
                    except json.JSONDecodeError:
                        pass

                return HandleLookupResult(
                    name=name,
                    handle=row['handle'],
                    confidence=row['confidence'],
                    status=LookupStatus(row['status']),
                    candidates=candidates,
                    cached=True
                )

            # Check negative cache
            cursor.execute("""
                SELECT reason
                FROM negative_cache
                WHERE name_normalized = ?
                AND (expires_at IS NULL OR expires_at > ?)
            """, (name_normalized, datetime.utcnow().isoformat()))

            row = cursor.fetchone()
            if row:
                return HandleLookupResult(
                    name=name,
                    handle=None,
                    confidence=0.0,
                    status=LookupStatus.NOT_FOUND,
                    cached=True
                )

        return None

    def cache_handle(
        self,
        name: str,
        result: HandleLookupResult,
        expiry_days: int = 30
    ):
        """
        Cache a handle lookup result.

        Args:
            name: Person name
            result: Lookup result to cache
            expiry_days: Days until cache expires
        """
        name_normalized = self._normalize_name(name)
        expires_at = datetime.utcnow() + timedelta(days=expiry_days)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Prepare sources JSON
            sources_json = json.dumps(
                [c.to_dict() for c in result.candidates],
                ensure_ascii=False
            ) if result.candidates else None

            # Insert or update cache
            cursor.execute("""
                INSERT INTO handle_cache
                (name, name_normalized, handle, confidence, status, sources_json, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name_normalized) DO UPDATE SET
                    handle = excluded.handle,
                    confidence = excluded.confidence,
                    status = excluded.status,
                    sources_json = excluded.sources_json,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
            """, (
                name,
                name_normalized,
                result.handle,
                result.confidence,
                result.status.value,
                sources_json,
                expires_at.isoformat(),
                datetime.utcnow().isoformat()
            ))

            logger.debug(f"Cached handle for {name}: {result.handle} (conf: {result.confidence:.2f})")

    def cache_negative(self, name: str, reason: str = "not_found", expiry_days: int = 7):
        """
        Cache a negative result (no handle found).

        Args:
            name: Person name
            reason: Why no handle was found
            expiry_days: Days until negative cache expires
        """
        name_normalized = self._normalize_name(name)
        expires_at = datetime.utcnow() + timedelta(days=expiry_days)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO negative_cache
                (name, name_normalized, reason, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name_normalized) DO UPDATE SET
                    lookup_count = lookup_count + 1,
                    last_lookup = CURRENT_TIMESTAMP,
                    reason = excluded.reason,
                    expires_at = excluded.expires_at
            """, (name, name_normalized, reason, expires_at.isoformat()))

            logger.debug(f"Cached negative result for {name}: {reason}")

    def invalidate_cache(self, name: str):
        """Remove a name from all caches."""
        name_normalized = self._normalize_name(name)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM handle_cache WHERE name_normalized = ?", (name_normalized,))
            cursor.execute("DELETE FROM negative_cache WHERE name_normalized = ?", (name_normalized,))
            logger.debug(f"Invalidated cache for {name}")

    # =========================================================================
    # Review Queue Operations
    # =========================================================================

    def add_to_review_queue(
        self,
        review_id: str,
        name: str,
        candidates: List[HandleCandidate],
        context: Optional[HandleLookupContext] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        priority: int = 0
    ) -> bool:
        """
        Add an item to the review queue.

        Args:
            review_id: Unique ID for this review
            name: Person name
            candidates: List of handle candidates
            context: Lookup context
            confidence_scores: Confidence scores by handle
            priority: Priority (higher = more urgent)

        Returns:
            True if added successfully
        """
        name_normalized = self._normalize_name(name)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if already in queue
            cursor.execute("""
                SELECT id FROM review_queue
                WHERE name_normalized = ? AND status = 'pending'
            """, (name_normalized,))

            if cursor.fetchone():
                logger.debug(f"{name} already in review queue")
                return False

            candidates_json = json.dumps(
                [c.to_dict() for c in candidates],
                ensure_ascii=False
            )
            context_json = context.to_json() if context else None
            scores_json = json.dumps(confidence_scores, ensure_ascii=False) if confidence_scores else None

            cursor.execute("""
                INSERT INTO review_queue
                (review_id, name, name_normalized, candidates_json, context_json,
                 confidence_scores_json, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                review_id, name, name_normalized, candidates_json,
                context_json, scores_json, priority
            ))

            logger.info(f"Added {name} to review queue (ID: {review_id})")
            return True

    def get_pending_reviews(self, limit: int = 50) -> List[ReviewQueueItem]:
        """
        Get pending review items.

        Args:
            limit: Maximum items to return

        Returns:
            List of review queue items
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT review_id, name, candidates_json, context_json,
                       confidence_scores_json, status, created_at
                FROM review_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
            """, (limit,))

            items = []
            for row in cursor.fetchall():
                candidates = []
                try:
                    for c_data in json.loads(row['candidates_json']):
                        candidates.append(HandleCandidate(
                            handle=c_data['handle'],
                            source=c_data['source'],
                            raw_confidence=c_data['raw_confidence'],
                            evidence=c_data.get('evidence', {}),
                            profile_data=c_data.get('profile_data')
                        ))
                except json.JSONDecodeError:
                    continue

                context = None
                if row['context_json']:
                    try:
                        context = HandleLookupContext.from_dict(json.loads(row['context_json']))
                    except json.JSONDecodeError:
                        pass

                confidence_scores = {}
                if row['confidence_scores_json']:
                    try:
                        confidence_scores = json.loads(row['confidence_scores_json'])
                    except json.JSONDecodeError:
                        pass

                items.append(ReviewQueueItem(
                    review_id=row['review_id'],
                    name=row['name'],
                    candidates=candidates,
                    context=context,
                    confidence_scores=confidence_scores,
                    status=row['status'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
                ))

            return items

    def approve_review(
        self,
        review_id: str,
        approved_handle: str,
        reviewer: str = "system",
        notes: str = ""
    ) -> bool:
        """
        Approve a review item and cache the result.

        Args:
            review_id: Review ID
            approved_handle: The approved handle
            reviewer: Who approved it
            notes: Optional notes

        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get the review item
            cursor.execute("""
                SELECT name FROM review_queue WHERE review_id = ?
            """, (review_id,))
            row = cursor.fetchone()

            if not row:
                logger.warning(f"Review {review_id} not found")
                return False

            name = row['name']

            # Update review status
            cursor.execute("""
                UPDATE review_queue
                SET status = 'approved',
                    approved_handle = ?,
                    reviewer = ?,
                    review_notes = ?,
                    reviewed_at = ?
                WHERE review_id = ?
            """, (approved_handle, reviewer, notes, datetime.utcnow().isoformat(), review_id))

            # Cache the approved handle
            result = HandleLookupResult(
                name=name,
                handle=approved_handle,
                confidence=1.0,  # Manual approval = full confidence
                status=LookupStatus.MANUAL_APPROVED
            )
            self.cache_handle(name, result)

            logger.info(f"Approved handle for {name}: @{approved_handle} (reviewer: {reviewer})")
            return True

    def reject_review(
        self,
        review_id: str,
        reviewer: str = "system",
        notes: str = ""
    ) -> bool:
        """
        Reject a review item.

        Args:
            review_id: Review ID
            reviewer: Who rejected it
            notes: Reason for rejection

        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE review_queue
                SET status = 'rejected',
                    reviewer = ?,
                    review_notes = ?,
                    reviewed_at = ?
                WHERE review_id = ?
            """, (reviewer, notes, datetime.utcnow().isoformat(), review_id))

            if cursor.rowcount > 0:
                logger.info(f"Rejected review {review_id} (reviewer: {reviewer})")
                return True
            return False

    # =========================================================================
    # Analytics
    # =========================================================================

    def log_lookup(
        self,
        name: str,
        result: HandleLookupResult,
        episode_id: Optional[str] = None
    ):
        """Log a lookup for analytics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO lookup_history
                (name, handle, confidence, status, sources_used, duration_ms, cached, episode_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                result.handle,
                result.confidence,
                result.status.value,
                ','.join(result.sources_used),
                result.lookup_time_ms,
                result.cached,
                episode_id
            ))

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Cache stats
            cursor.execute("SELECT COUNT(*) FROM handle_cache")
            stats['cached_handles'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM negative_cache")
            stats['negative_cache'] = cursor.fetchone()[0]

            # Review queue stats
            cursor.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'")
            stats['pending_reviews'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'approved'")
            stats['approved_reviews'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'rejected'")
            stats['rejected_reviews'] = cursor.fetchone()[0]

            # Lookup history stats (last 24 hours)
            yesterday = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            cursor.execute("""
                SELECT COUNT(*), AVG(duration_ms)
                FROM lookup_history
                WHERE created_at > ?
            """, (yesterday,))
            row = cursor.fetchone()
            stats['lookups_24h'] = row[0]
            stats['avg_lookup_ms_24h'] = round(row[1], 2) if row[1] else 0

            cursor.execute("""
                SELECT COUNT(*)
                FROM lookup_history
                WHERE created_at > ? AND cached = 1
            """, (yesterday,))
            stats['cache_hits_24h'] = cursor.fetchone()[0]

            return stats

    def cleanup_expired(self) -> int:
        """Remove expired cache entries. Returns count of removed entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            cursor.execute("DELETE FROM handle_cache WHERE expires_at < ?", (now,))
            positive_removed = cursor.rowcount

            cursor.execute("DELETE FROM negative_cache WHERE expires_at < ?", (now,))
            negative_removed = cursor.rowcount

            total = positive_removed + negative_removed
            if total > 0:
                logger.info(f"Cleaned up {total} expired cache entries")
            return total
