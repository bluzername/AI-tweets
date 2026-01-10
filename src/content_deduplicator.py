#!/usr/bin/env python3
"""
Content Deduplication System - Detect and prevent similar/duplicate content.
Uses semantic embeddings to identify similar tweets and avoid repetition.
"""

import logging
import os
import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ContentRecord:
    """Record of posted or scheduled content."""
    content_id: str
    text: str
    embedding: Optional[List[float]]
    posted_at: str
    podcast_name: str
    episode_id: Optional[str]
    category: str
    similarity_hash: str  # Quick hash for exact duplicates


class ContentDeduplicator:
    """
    Detect and prevent duplicate or overly similar content.

    Features:
    - Exact duplicate detection (hash-based)
    - Semantic similarity detection (embedding-based)
    - Topic clustering to avoid same topics
    - Time-based deduplication windows
    - Content history tracking
    """

    def __init__(self,
                 db_path: str = "data/content_history.db",
                 openai_api_key: Optional[str] = None,
                 similarity_threshold: float = 0.85,
                 time_window_days: int = 14):
        """
        Initialize content deduplicator.

        Args:
            db_path: Path to SQLite database for content history
            openai_api_key: OpenAI API key for embeddings
            similarity_threshold: Cosine similarity threshold (0.0-1.0)
            time_window_days: Check for duplicates within this many days
        """

        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.time_window_days = time_window_days

        # OpenAI for embeddings
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

        if self.openai_api_key and self.openai_api_key != "your_openai_api_key_here":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                logger.info("✅ Deduplicator initialized with embeddings")
            except Exception as e:
                logger.warning(f"⚠️ Embeddings unavailable: {e}")

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize content history database."""

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_history (
                content_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT,  -- JSON array of floats
                posted_at TEXT NOT NULL,
                podcast_name TEXT,
                episode_id TEXT,
                category TEXT,
                similarity_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_posted_at
            ON content_history(posted_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_similarity_hash
            ON content_history(similarity_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_podcast_name
            ON content_history(podcast_name)
        """)

        conn.commit()
        conn.close()

        logger.info(f"✅ Content history database initialized: {self.db_path}")

    def _compute_similarity_hash(self, text: str) -> str:
        """Compute a quick hash for exact duplicate detection."""

        # Normalize text
        normalized = text.lower().strip()
        normalized = ' '.join(normalized.split())  # Normalize whitespace

        # MD5 hash
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get text embedding from OpenAI."""

        if not self.client:
            return None

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # Fast and cheap
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""

        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def check_duplicate(self, text: str,
                       podcast_name: Optional[str] = None,
                       category: Optional[str] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if content is a duplicate or too similar to recent content.

        Args:
            text: Content text to check
            podcast_name: Optional podcast filter
            category: Optional category filter

        Returns:
            Tuple of (is_duplicate, similar_content_info)
        """

        # 1. Check exact duplicate via hash
        similarity_hash = self._compute_similarity_hash(text)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check for exact match in recent history
        cutoff_date = (datetime.utcnow() - timedelta(days=self.time_window_days)).isoformat()

        cursor.execute("""
            SELECT content_id, text, posted_at, podcast_name
            FROM content_history
            WHERE similarity_hash = ?
            AND posted_at >= ?
            LIMIT 1
        """, (similarity_hash, cutoff_date))

        exact_match = cursor.fetchone()

        if exact_match:
            conn.close()
            logger.warning(f"⚠️ Exact duplicate found: {exact_match[0]}")
            return True, {
                "type": "exact",
                "content_id": exact_match[0],
                "text": exact_match[1],
                "posted_at": exact_match[2],
                "similarity": 1.0
            }

        # 2. Check semantic similarity via embeddings
        if self.client:
            embedding = self._get_embedding(text)

            if embedding:
                # Get recent content
                cursor.execute("""
                    SELECT content_id, text, embedding, posted_at, podcast_name, category
                    FROM content_history
                    WHERE posted_at >= ?
                    AND embedding IS NOT NULL
                """, (cutoff_date,))

                recent_content = cursor.fetchall()

                for record in recent_content:
                    content_id, stored_text, stored_embedding_json, posted_at, stored_podcast, stored_category = record

                    try:
                        stored_embedding = json.loads(stored_embedding_json)
                    except:
                        continue

                    similarity = self._cosine_similarity(embedding, stored_embedding)

                    if similarity >= self.similarity_threshold:
                        conn.close()
                        logger.warning(f"⚠️ Similar content found: {content_id} (similarity: {similarity:.2f})")
                        return True, {
                            "type": "similar",
                            "content_id": content_id,
                            "text": stored_text,
                            "posted_at": posted_at,
                            "similarity": similarity,
                            "podcast_name": stored_podcast,
                            "category": stored_category
                        }

        conn.close()

        # No duplicates found
        return False, None

    def add_content(self,
                   text: str,
                   posted_at: Optional[str] = None,
                   podcast_name: Optional[str] = None,
                   episode_id: Optional[str] = None,
                   category: Optional[str] = None) -> str:
        """
        Add content to history for future deduplication checks.

        Args:
            text: Content text
            posted_at: When posted (ISO format, defaults to now)
            podcast_name: Source podcast
            episode_id: Source episode
            category: Content category

        Returns:
            Content ID
        """

        # Generate content ID
        content_id = hashlib.sha256(
            f"{text}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        posted_at = posted_at or datetime.utcnow().isoformat()

        # Compute hash
        similarity_hash = self._compute_similarity_hash(text)

        # Get embedding
        embedding = self._get_embedding(text)
        embedding_json = json.dumps(embedding) if embedding else None

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO content_history
            (content_id, text, embedding, posted_at, podcast_name, episode_id, category, similarity_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (content_id, text, embedding_json, posted_at, podcast_name, episode_id, category, similarity_hash))

        conn.commit()
        conn.close()

        logger.info(f"✅ Content added to history: {content_id}")

        return content_id

    def filter_duplicates(self,
                         content_list: List[Dict[str, Any]],
                         text_field: str = "text") -> List[Dict[str, Any]]:
        """
        Filter a list of content items, removing duplicates.

        Args:
            content_list: List of content dictionaries
            text_field: Field name containing the text

        Returns:
            Filtered list with duplicates removed
        """

        filtered = []

        for item in content_list:
            text = item.get(text_field, "")

            if not text:
                continue

            is_duplicate, match_info = self.check_duplicate(
                text,
                podcast_name=item.get("podcast_name"),
                category=item.get("category")
            )

            if is_duplicate:
                logger.info(f"⏭️ Skipping duplicate: {text[:50]}...")
                logger.info(f"   Similar to: {match_info['content_id']} (similarity: {match_info.get('similarity', 1.0):.2f})")
            else:
                filtered.append(item)

        logger.info(f"✅ Filtered: {len(content_list)} → {len(filtered)} (removed {len(content_list) - len(filtered)} duplicates)")

        return filtered

    def get_topic_coverage(self, days: int = 7) -> Dict[str, int]:
        """
        Get topic coverage statistics for recent content.

        Args:
            days: Look back this many days

        Returns:
            Dictionary of topic counts
        """

        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT podcast_name, category, COUNT(*)
            FROM content_history
            WHERE posted_at >= ?
            GROUP BY podcast_name, category
        """, (cutoff_date,))

        results = cursor.fetchall()
        conn.close()

        coverage = {}
        for podcast, category, count in results:
            key = f"{podcast}:{category}" if podcast and category else podcast or category or "unknown"
            coverage[key] = count

        return coverage

    def cleanup_old_content(self, days: int = 90):
        """
        Remove content older than specified days.

        Args:
            days: Remove content older than this
        """

        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM content_history
            WHERE posted_at < ?
        """, (cutoff_date,))

        deleted_count = cursor.rowcount

        conn.commit()
        conn.close()

        logger.info(f"🗑️ Cleaned up {deleted_count} old content records")

    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total content
        cursor.execute("SELECT COUNT(*) FROM content_history")
        total = cursor.fetchone()[0]

        # Recent content (last 7 days)
        cutoff_7d = (datetime.utcnow() - timedelta(days=7)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM content_history WHERE posted_at >= ?", (cutoff_7d,))
        recent_7d = cursor.fetchone()[0]

        # With embeddings
        cursor.execute("SELECT COUNT(*) FROM content_history WHERE embedding IS NOT NULL")
        with_embeddings = cursor.fetchone()[0]

        # By podcast
        cursor.execute("""
            SELECT podcast_name, COUNT(*)
            FROM content_history
            GROUP BY podcast_name
            ORDER BY COUNT(*) DESC
            LIMIT 5
        """)
        top_podcasts = dict(cursor.fetchall())

        conn.close()

        return {
            "total_content": total,
            "recent_7_days": recent_7d,
            "with_embeddings": with_embeddings,
            "embedding_coverage": (with_embeddings / total * 100) if total > 0 else 0,
            "top_podcasts": top_podcasts,
            "similarity_threshold": self.similarity_threshold,
            "time_window_days": self.time_window_days
        }


def main():
    """CLI for content deduplication."""
    import argparse

    parser = argparse.ArgumentParser(description="Content Deduplication System")
    parser.add_argument("--check", type=str, help="Check if text is duplicate")
    parser.add_argument("--add", type=str, help="Add text to history")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--cleanup", type=int, help="Clean content older than N days")
    parser.add_argument("--coverage", action="store_true", help="Show topic coverage")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    deduplicator = ContentDeduplicator()

    if args.check:
        is_dup, info = deduplicator.check_duplicate(args.check)

        if is_dup:
            print(f"❌ DUPLICATE DETECTED ({info['type']})")
            print(f"   Similar to: {info['content_id']}")
            print(f"   Posted: {info['posted_at']}")
            print(f"   Similarity: {info.get('similarity', 1.0):.2f}")
        else:
            print("✅ NO DUPLICATE - Content is unique")

    elif args.add:
        content_id = deduplicator.add_content(
            text=args.add,
            podcast_name="Test Podcast",
            category="test"
        )
        print(f"✅ Added to history: {content_id}")

    elif args.stats:
        stats = deduplicator.get_statistics()
        print(json.dumps(stats, indent=2))

    elif args.cleanup:
        deduplicator.cleanup_old_content(days=args.cleanup)
        print(f"✅ Cleaned up content older than {args.cleanup} days")

    elif args.coverage:
        coverage = deduplicator.get_topic_coverage(days=7)
        print("\n📊 Topic Coverage (last 7 days):")
        for topic, count in sorted(coverage.items(), key=lambda x: x[1], reverse=True):
            print(f"  {topic}: {count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
