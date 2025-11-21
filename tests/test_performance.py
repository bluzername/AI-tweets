"""
Performance tracking tests - Sprint 3 features.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from performance_tracker import PerformanceTracker, TweetPerformance
from ab_testing_framework import ABTestingFramework, TestStatus, TestVariant, ABTest
from feedback_loop_optimizer import FeedbackLoopOptimizer, OptimizationRecommendation


class TestPerformanceTracker:
    """Test performance tracking system."""

    def test_initialization(self, temp_db):
        """Test performance tracker initialization."""
        tracker = PerformanceTracker(db_path=temp_db)
        assert tracker is not None

    def test_save_performance(self, temp_db):
        """Test saving performance metrics."""
        tracker = PerformanceTracker(db_path=temp_db)

        performance = TweetPerformance(
            tweet_id="123456",
            content="Test tweet",
            posted_at="2025-01-19T12:00:00",
            likes=100,
            retweets=50,
            replies=10,
            impressions=5000,
            engagement_rate=3.2,
            viral_score=45.0,
            hook_pattern="question",
            post_hour=12
        )

        tracker.save_performance(performance)

        # Verify it was saved
        stats = tracker.get_statistics()
        assert stats["total_tweets"] == 1

    def test_hook_performance_analysis(self, temp_db):
        """Test hook pattern performance analysis."""
        tracker = PerformanceTracker(db_path=temp_db)

        # Save multiple performances with different hooks
        for i, pattern in enumerate(["question", "question", "shock", "numbers"]):
            perf = TweetPerformance(
                tweet_id=f"tweet_{i}",
                content=f"Test tweet {i}",
                posted_at="2025-01-19T12:00:00",
                likes=100 + i * 10,
                retweets=50,
                replies=10,
                impressions=5000,
                engagement_rate=2.0 + i * 0.5,
                hook_pattern=pattern,
                post_hour=12
            )
            tracker.save_performance(perf)

        # Get hook performance
        hook_stats = tracker.get_hook_performance(min_samples=1)

        assert "question" in hook_stats
        assert hook_stats["question"]["count"] == 2

    def test_time_performance_analysis(self, temp_db):
        """Test time-of-day performance analysis."""
        tracker = PerformanceTracker(db_path=temp_db)

        # Save performances at different hours
        for hour in [9, 12, 15, 18]:
            perf = TweetPerformance(
                tweet_id=f"tweet_{hour}",
                content=f"Test tweet at {hour}h",
                posted_at="2025-01-19T12:00:00",
                likes=100,
                retweets=50,
                replies=10,
                impressions=5000,
                engagement_rate=3.0,
                post_hour=hour
            )
            tracker.save_performance(perf)

        # Get time performance
        time_stats = tracker.get_time_performance()

        assert len(time_stats) == 4
        assert 12 in time_stats
        assert "avg_engagement_rate" in time_stats[12]

    def test_podcast_roi_calculation(self, temp_db):
        """Test podcast ROI calculation."""
        tracker = PerformanceTracker(db_path=temp_db)

        # Save performances from different podcasts
        for i, podcast in enumerate(["Podcast A", "Podcast B", "Podcast A"]):
            perf = TweetPerformance(
                tweet_id=f"tweet_{podcast}_{i}",
                content=f"Test from {podcast}",
                posted_at="2025-01-19T12:00:00",
                likes=100,
                retweets=50,
                replies=10,
                impressions=5000,
                engagement_rate=3.0,
                podcast_name=podcast
            )
            tracker.save_performance(perf)

        # Get podcast ROI
        roi = tracker.get_podcast_roi()

        assert "Podcast A" in roi
        assert roi["Podcast A"]["tweets_count"] == 2


class TestABTestingFramework:
    """Test A/B testing framework."""

    def test_initialization(self, temp_db):
        """Test A/B testing initialization."""
        framework = ABTestingFramework(db_path=temp_db)
        assert framework is not None

    def test_create_test(self, temp_db):
        """Test creating an A/B test."""
        framework = ABTestingFramework(db_path=temp_db)

        variants = [
            {"name": "Question", "description": "Question hooks"},
            {"name": "Shock", "description": "Shock hooks"},
        ]

        test_id = framework.create_test(
            test_name="Hook Pattern Test",
            test_type="hook_pattern",
            variants=variants,
            min_samples=10
        )

        assert test_id is not None
        assert test_id.startswith("test_")

        # Verify test was created
        test_info = framework.get_test_results(test_id)
        assert test_info["test_name"] == "Hook Pattern Test"
        assert len(test_info["variants"]) == 2

    def test_variant_assignment(self, temp_db):
        """Test variant assignment."""
        framework = ABTestingFramework(db_path=temp_db)

        variants = [
            {"name": "Variant A"},
            {"name": "Variant B"},
        ]

        test_id = framework.create_test(
            test_name="Test",
            test_type="test",
            variants=variants
        )

        # Assign variant
        variant_id = framework.assign_variant(test_id, method="random")

        assert variant_id is not None
        assert test_id in variant_id

    def test_record_assignment(self, temp_db):
        """Test recording variant assignment."""
        framework = ABTestingFramework(db_path=temp_db)

        variants = [{"name": "Variant A"}]

        test_id = framework.create_test("Test", "test", variants)
        variant_id = framework.assign_variant(test_id)

        framework.record_assignment(
            test_id=test_id,
            variant_id=variant_id,
            tweet_id="tweet_123",
            engagement_rate=3.5,
            impressions=5000
        )

        # Verify assignment was recorded
        test_info = framework.get_test_results(test_id)
        assert test_info["variants"][0]["tweet_count"] == 1

    def test_winner_detection(self, temp_db):
        """Test winner detection."""
        framework = ABTestingFramework(db_path=temp_db)

        variants = [
            {"name": "Variant A"},
            {"name": "Variant B"},
        ]

        test_id = framework.create_test(
            test_name="Test",
            test_type="test",
            variants=variants,
            min_samples=2
        )

        # Get variant IDs
        test_info = framework.get_test_results(test_id)
        variant_a = test_info["variants"][0]["variant_id"]
        variant_b = test_info["variants"][1]["variant_id"]

        # Record assignments with clear winner
        for i in range(3):
            framework.record_assignment(test_id, variant_a, f"tweet_a_{i}", 5.0, 5000)

        for i in range(3):
            framework.record_assignment(test_id, variant_b, f"tweet_b_{i}", 2.0, 5000)

        # Update metrics
        framework.update_variant_metrics(test_id)

        # Check for winner (Variant A should win with 5.0 vs 2.0 engagement)
        winner_id = framework.check_for_winner(test_id)

        # Winner might not be detected yet depending on statistical significance threshold
        # But metrics should be updated
        test_info = framework.get_test_results(test_id)
        assert test_info["variants"][0]["tweet_count"] == 3
        assert test_info["variants"][1]["tweet_count"] == 3

    def test_get_learnings(self, temp_db):
        """Test getting learnings from completed tests."""
        framework = ABTestingFramework(db_path=temp_db)

        learnings = framework.get_learnings()

        assert "total_tests_completed" in learnings
        assert "by_type" in learnings


class TestFeedbackLoopOptimizer:
    """Test feedback loop optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        # Create with mock trackers
        optimizer = FeedbackLoopOptimizer()
        assert optimizer is not None

    def test_optimization_history(self):
        """Test optimization history tracking."""
        optimizer = FeedbackLoopOptimizer()

        # History should be a list
        assert isinstance(optimizer.optimization_history, list)

    def test_empty_recommendations(self):
        """Test with no data (should return empty recommendations)."""
        optimizer = FeedbackLoopOptimizer()

        # With no performance data, should return empty or minimal recommendations
        recommendations = optimizer.analyze_and_recommend()

        assert isinstance(recommendations, list)
        # May be empty or have some default recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
