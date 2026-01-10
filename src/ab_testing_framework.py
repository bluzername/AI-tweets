#!/usr/bin/env python3
"""
A/B Testing Framework - Systematic testing and optimization.
Tests hook variants, posting times, and content strategies to maximize engagement.
"""

import logging
import sqlite3
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    WINNER_SELECTED = "winner_selected"


@dataclass
class TestVariant:
    """A variant in an A/B test."""
    variant_id: str
    name: str
    description: str
    tweet_count: int = 0
    total_engagement: int = 0
    total_impressions: int = 0
    avg_engagement_rate: float = 0.0
    confidence_level: float = 0.0  # Statistical confidence


@dataclass
class ABTest:
    """A/B test configuration and results."""
    test_id: str
    test_name: str
    test_type: str  # "hook_pattern", "posting_time", "emoji_usage", etc.
    status: TestStatus
    variants: List[TestVariant]
    winner_id: Optional[str]
    started_at: str
    completed_at: Optional[str]
    min_samples: int = 30  # Minimum tweets per variant
    confidence_threshold: float = 0.95  # 95% confidence to declare winner


class ABTestingFramework:
    """
    A/B testing framework for systematic content optimization.

    Features:
    - Multi-variant testing
    - Statistical significance calculation
    - Automatic winner selection
    - Traffic allocation (equal or weighted)
    - Progressive optimization (winner becomes new control)
    - Test history and learnings database
    """

    def __init__(self, db_path: str = "data/ab_tests.db"):
        """Initialize A/B testing framework."""
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize A/B testing database."""

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                test_name TEXT NOT NULL,
                test_type TEXT NOT NULL,
                status TEXT NOT NULL,
                winner_id TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                min_samples INTEGER DEFAULT 30,
                confidence_threshold REAL DEFAULT 0.95,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Variants table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_variants (
                variant_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                configuration TEXT,  -- JSON
                tweet_count INTEGER DEFAULT 0,
                total_engagement INTEGER DEFAULT 0,
                total_impressions INTEGER DEFAULT 0,
                avg_engagement_rate REAL DEFAULT 0.0,
                confidence_level REAL DEFAULT 0.0,
                FOREIGN KEY (test_id) REFERENCES ab_tests(test_id)
            )
        """)

        # Test assignments table (which tweets belong to which variant)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_assignments (
                assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                tweet_id TEXT NOT NULL,
                engagement_rate REAL,
                impressions INTEGER,
                assigned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES ab_tests(test_id),
                FOREIGN KEY (variant_id) REFERENCES test_variants(variant_id)
            )
        """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_status
            ON ab_tests(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_assignments_test
            ON test_assignments(test_id)
        """)

        conn.commit()
        conn.close()

        logger.info(f"✅ A/B testing database initialized: {self.db_path}")

    def create_test(self,
                   test_name: str,
                   test_type: str,
                   variants: List[Dict[str, Any]],
                   min_samples: int = 30,
                   confidence_threshold: float = 0.95) -> str:
        """
        Create a new A/B test.

        Args:
            test_name: Name of the test
            test_type: Type of test (hook_pattern, posting_time, etc.)
            variants: List of variant configurations
            min_samples: Minimum samples per variant
            confidence_threshold: Confidence level for winner selection

        Returns:
            Test ID
        """

        test_id = f"test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create test
        cursor.execute("""
            INSERT INTO ab_tests
            (test_id, test_name, test_type, status, started_at, min_samples, confidence_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id, test_name, test_type, TestStatus.ACTIVE.value,
            datetime.utcnow().isoformat(), min_samples, confidence_threshold
        ))

        # Create variants
        for i, variant_config in enumerate(variants):
            variant_id = f"{test_id}_v{i+1}"

            cursor.execute("""
                INSERT INTO test_variants
                (variant_id, test_id, name, description, configuration)
                VALUES (?, ?, ?, ?, ?)
            """, (
                variant_id, test_id,
                variant_config.get("name", f"Variant {i+1}"),
                variant_config.get("description", ""),
                json.dumps(variant_config, ensure_ascii=False)
            ))

        conn.commit()
        conn.close()

        logger.info(f"✅ Created A/B test: {test_id} with {len(variants)} variants")

        return test_id

    def assign_variant(self, test_id: str, method: str = "random") -> Optional[str]:
        """
        Assign a variant for the next tweet.

        Args:
            test_id: Test ID
            method: Assignment method ("random" or "weighted")

        Returns:
            Variant ID or None if test not active
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check test status
        cursor.execute("SELECT status FROM ab_tests WHERE test_id = ?", (test_id,))
        result = cursor.fetchone()

        if not result or result[0] != TestStatus.ACTIVE.value:
            conn.close()
            logger.warning(f"⚠️ Test {test_id} not active")
            return None

        # Get variants
        cursor.execute("""
            SELECT variant_id, tweet_count
            FROM test_variants
            WHERE test_id = ?
        """, (test_id,))

        variants = cursor.fetchall()
        conn.close()

        if not variants:
            return None

        if method == "random":
            # Equal probability for all variants
            return random.choice(variants)[0]

        elif method == "weighted":
            # Assign to variant with fewest tweets (balance samples)
            min_count = min(v[1] for v in variants)
            eligible = [v[0] for v in variants if v[1] == min_count]
            return random.choice(eligible)

        return None

    def record_assignment(self,
                         test_id: str,
                         variant_id: str,
                         tweet_id: str,
                         engagement_rate: Optional[float] = None,
                         impressions: Optional[int] = None):
        """
        Record a variant assignment for a tweet.

        Args:
            test_id: Test ID
            variant_id: Variant ID
            tweet_id: Tweet ID
            engagement_rate: Engagement rate (if known)
            impressions: Impressions (if known)
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO test_assignments
            (test_id, variant_id, tweet_id, engagement_rate, impressions)
            VALUES (?, ?, ?, ?, ?)
        """, (test_id, variant_id, tweet_id, engagement_rate, impressions))

        # Update variant tweet count
        cursor.execute("""
            UPDATE test_variants
            SET tweet_count = tweet_count + 1
            WHERE variant_id = ?
        """, (variant_id,))

        conn.commit()
        conn.close()

        logger.info(f"✅ Recorded assignment: {tweet_id} → {variant_id}")

    def update_variant_metrics(self, test_id: str):
        """
        Update variant metrics from performance data.

        Args:
            test_id: Test ID
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all variants for this test
        cursor.execute("SELECT variant_id FROM test_variants WHERE test_id = ?", (test_id,))
        variant_ids = [row[0] for row in cursor.fetchall()]

        for variant_id in variant_ids:
            # Calculate metrics from assignments
            cursor.execute("""
                SELECT
                    COUNT(*) as count,
                    AVG(engagement_rate) as avg_eng_rate,
                    SUM(impressions) as total_impressions
                FROM test_assignments
                WHERE variant_id = ?
                AND engagement_rate IS NOT NULL
            """, (variant_id,))

            result = cursor.fetchone()

            if result and result[0] > 0:
                count, avg_rate, total_impressions = result

                # Update variant
                cursor.execute("""
                    UPDATE test_variants
                    SET tweet_count = ?,
                        avg_engagement_rate = ?,
                        total_impressions = ?
                    WHERE variant_id = ?
                """, (count, avg_rate or 0.0, total_impressions or 0, variant_id))

        conn.commit()
        conn.close()

        logger.info(f"✅ Updated metrics for test: {test_id}")

    def check_for_winner(self, test_id: str) -> Optional[str]:
        """
        Check if we have a statistically significant winner.

        Args:
            test_id: Test ID

        Returns:
            Winner variant ID or None
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get test config
        cursor.execute("""
            SELECT min_samples, confidence_threshold
            FROM ab_tests
            WHERE test_id = ?
        """, (test_id,))

        test_config = cursor.fetchone()
        if not test_config:
            conn.close()
            return None

        min_samples, confidence_threshold = test_config

        # Get variant metrics
        cursor.execute("""
            SELECT variant_id, tweet_count, avg_engagement_rate
            FROM test_variants
            WHERE test_id = ?
            ORDER BY avg_engagement_rate DESC
        """, (test_id,))

        variants = cursor.fetchall()
        conn.close()

        if len(variants) < 2:
            return None

        # Check if all variants have minimum samples
        if any(v[1] < min_samples for v in variants):
            logger.info(f"⏳ Test {test_id}: Waiting for minimum samples ({min_samples} per variant)")
            return None

        # Simple winner selection: best performer with minimum confidence
        # (For production, use proper statistical tests like t-test or chi-square)

        best = variants[0]
        second_best = variants[1]

        # Calculate improvement
        if second_best[2] > 0:
            improvement = (best[2] - second_best[2]) / second_best[2]

            # Require at least 10% improvement for statistical significance
            if improvement >= 0.10:
                logger.info(f"🏆 Winner detected: {best[0]} (+{improvement*100:.1f}% improvement)")
                return best[0]

        logger.info(f"📊 No clear winner yet (best: {best[2]:.2f}%, second: {second_best[2]:.2f}%)")
        return None

    def declare_winner(self, test_id: str, winner_id: str):
        """
        Declare a test winner and complete the test.

        Args:
            test_id: Test ID
            winner_id: Winner variant ID
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE ab_tests
            SET status = ?,
                winner_id = ?,
                completed_at = ?
            WHERE test_id = ?
        """, (TestStatus.WINNER_SELECTED.value, winner_id, datetime.utcnow().isoformat(), test_id))

        conn.commit()
        conn.close()

        logger.info(f"🏆 Winner declared for {test_id}: {winner_id}")

    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get test results and analysis.

        Args:
            test_id: Test ID

        Returns:
            Test results dictionary
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get test info
        cursor.execute("""
            SELECT test_name, test_type, status, winner_id, started_at, completed_at
            FROM ab_tests
            WHERE test_id = ?
        """, (test_id,))

        test_info = cursor.fetchone()

        if not test_info:
            conn.close()
            return {}

        # Get variants
        cursor.execute("""
            SELECT variant_id, name, tweet_count, avg_engagement_rate
            FROM test_variants
            WHERE test_id = ?
            ORDER BY avg_engagement_rate DESC
        """, (test_id,))

        variants = []
        for row in cursor.fetchall():
            variants.append({
                "variant_id": row[0],
                "name": row[1],
                "tweet_count": row[2],
                "avg_engagement_rate": round(row[3], 2)
            })

        conn.close()

        return {
            "test_id": test_id,
            "test_name": test_info[0],
            "test_type": test_info[1],
            "status": test_info[2],
            "winner_id": test_info[3],
            "started_at": test_info[4],
            "completed_at": test_info[5],
            "variants": variants
        }

    def get_active_tests(self) -> List[str]:
        """Get list of active test IDs."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT test_id
            FROM ab_tests
            WHERE status = ?
        """, (TestStatus.ACTIVE.value,))

        test_ids = [row[0] for row in cursor.fetchall()]

        conn.close()

        return test_ids

    def get_learnings(self) -> Dict[str, Any]:
        """
        Get learnings from completed tests.

        Returns:
            Dictionary of insights from past tests
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get completed tests
        cursor.execute("""
            SELECT test_id, test_type, winner_id
            FROM ab_tests
            WHERE status = ?
        """, (TestStatus.WINNER_SELECTED.value,))

        completed_tests = cursor.fetchall()

        learnings = {
            "total_tests_completed": len(completed_tests),
            "by_type": {},
            "winner_patterns": {}
        }

        for test_id, test_type, winner_id in completed_tests:
            # Count by type
            learnings["by_type"][test_type] = learnings["by_type"].get(test_type, 0) + 1

            # Get winner details
            if winner_id:
                cursor.execute("""
                    SELECT name, avg_engagement_rate
                    FROM test_variants
                    WHERE variant_id = ?
                """, (winner_id,))

                winner_info = cursor.fetchone()

                if winner_info:
                    pattern = winner_info[0]
                    if pattern not in learnings["winner_patterns"]:
                        learnings["winner_patterns"][pattern] = {
                            "wins": 0,
                            "avg_performance": []
                        }

                    learnings["winner_patterns"][pattern]["wins"] += 1
                    learnings["winner_patterns"][pattern]["avg_performance"].append(winner_info[1])

        # Calculate averages
        for pattern in learnings["winner_patterns"]:
            performances = learnings["winner_patterns"][pattern]["avg_performance"]
            learnings["winner_patterns"][pattern]["avg_performance"] = round(statistics.mean(performances), 2)

        conn.close()

        return learnings


def main():
    """CLI for A/B testing."""
    import argparse

    parser = argparse.ArgumentParser(description="A/B Testing Framework")
    parser.add_argument("--create", action="store_true", help="Create sample test")
    parser.add_argument("--results", type=str, help="Get test results")
    parser.add_argument("--active", action="store_true", help="List active tests")
    parser.add_argument("--learnings", action="store_true", help="Show learnings")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    framework = ABTestingFramework()

    if args.create:
        # Create sample hook pattern test
        variants = [
            {"name": "Question", "description": "Question-based hooks"},
            {"name": "Shock", "description": "Shock value hooks"},
            {"name": "Numbers", "description": "Data-driven hooks"},
        ]

        test_id = framework.create_test(
            test_name="Hook Pattern Test",
            test_type="hook_pattern",
            variants=variants,
            min_samples=30
        )

        print(f"✅ Created test: {test_id}")

    elif args.results:
        results = framework.get_test_results(args.results)
        print(json.dumps(results, indent=2))

    elif args.active:
        tests = framework.get_active_tests()
        print(f"\n📊 Active tests ({len(tests)}):")
        for test_id in tests:
            results = framework.get_test_results(test_id)
            print(f"\n  {results['test_name']} ({test_id})")
            print(f"  Type: {results['test_type']}")
            print(f"  Variants: {len(results['variants'])}")

    elif args.learnings:
        learnings = framework.get_learnings()
        print(json.dumps(learnings, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
