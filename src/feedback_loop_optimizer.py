#!/usr/bin/env python3
"""
Feedback Loop Optimizer - Learn from performance and auto-improve.
Uses performance data to automatically optimize viral scores, hook selection,
posting times, and content strategies.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from src.performance_tracker import PerformanceTracker
from src.ab_testing_framework import ABTestingFramework

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecommendation:
    """A recommended optimization based on data."""
    category: str  # "hook_pattern", "posting_time", "podcast_source", etc.
    action: str  # "increase", "decrease", "replace", "test"
    target: str  # What to optimize
    current_value: Any
    recommended_value: Any
    confidence: float  # 0.0-1.0
    expected_improvement: float  # Percentage improvement expected
    reasoning: str


class FeedbackLoopOptimizer:
    """
    Automated optimization based on performance feedback.

    Features:
    - Auto-adjusting viral score weights
    - Hook pattern optimization
    - Posting time optimization
    - Podcast source prioritization
    - Content strategy refinement
    - Continuous A/B testing
    """

    def __init__(self,
                 performance_tracker: Optional[PerformanceTracker] = None,
                 ab_testing: Optional[ABTestingFramework] = None):
        """
        Initialize feedback loop optimizer.

        Args:
            performance_tracker: Performance tracker instance
            ab_testing: A/B testing framework instance
        """

        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.ab_testing = ab_testing or ABTestingFramework()

        # Optimization state
        self.optimization_history = self._load_optimization_history()

    def _load_optimization_history(self) -> List[Dict[str, Any]]:
        """Load optimization history from file."""

        history_file = Path("data/optimization_history.json")

        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load optimization history: {e}")

        return []

    def _save_optimization_history(self):
        """Save optimization history to file."""

        history_file = Path("data/optimization_history.json")
        Path(history_file).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_history[-100:], f, indent=2, ensure_ascii=False)  # Keep last 100
        except Exception as e:
            logger.warning(f"Could not save optimization history: {e}")

    def analyze_and_recommend(self) -> List[OptimizationRecommendation]:
        """
        Analyze performance data and generate optimization recommendations.

        Returns:
            List of optimization recommendations
        """

        recommendations = []

        # 1. Analyze hook patterns
        hook_recs = self._analyze_hook_patterns()
        recommendations.extend(hook_recs)

        # 2. Analyze posting times
        time_recs = self._analyze_posting_times()
        recommendations.extend(time_recs)

        # 3. Analyze podcast sources
        podcast_recs = self._analyze_podcast_sources()
        recommendations.extend(podcast_recs)

        # 4. Check A/B test results
        ab_recs = self._analyze_ab_tests()
        recommendations.extend(ab_recs)

        # Sort by expected improvement
        recommendations.sort(key=lambda x: x.expected_improvement, reverse=True)

        logger.info(f"📊 Generated {len(recommendations)} optimization recommendations")

        return recommendations

    def _analyze_hook_patterns(self) -> List[OptimizationRecommendation]:
        """Analyze hook pattern performance and recommend changes."""

        recommendations = []

        try:
            # Get hook performance stats
            hook_stats = self.performance_tracker.get_hook_performance(min_samples=5)

            if not hook_stats:
                return recommendations

            # Find best and worst performers
            patterns_by_performance = sorted(
                hook_stats.items(),
                key=lambda x: x[1]["avg_engagement_rate"],
                reverse=True
            )

            if len(patterns_by_performance) >= 2:
                best = patterns_by_performance[0]
                worst = patterns_by_performance[-1]

                # Recommend using more of the best pattern
                improvement = (best[1]["avg_engagement_rate"] - worst[1]["avg_engagement_rate"]) / worst[1]["avg_engagement_rate"] * 100

                if improvement > 20:  # 20% better
                    recommendations.append(
                        OptimizationRecommendation(
                            category="hook_pattern",
                            action="increase",
                            target=best[0],
                            current_value=f"{best[1]['count']} uses",
                            recommended_value="Increase usage by 50%",
                            confidence=0.8 if best[1]['count'] >= 10 else 0.6,
                            expected_improvement=improvement,
                            reasoning=f"{best[0]} performs {improvement:.1f}% better than {worst[0]}"
                        )
                    )

                    # Recommend reducing worst pattern
                    recommendations.append(
                        OptimizationRecommendation(
                            category="hook_pattern",
                            action="decrease",
                            target=worst[0],
                            current_value=f"{worst[1]['count']} uses",
                            recommended_value="Reduce usage by 50%",
                            confidence=0.7,
                            expected_improvement=improvement / 2,
                            reasoning=f"{worst[0]} underperforms by {improvement:.1f}%"
                        )
                    )

        except Exception as e:
            logger.error(f"Hook pattern analysis failed: {e}")

        return recommendations

    def _analyze_posting_times(self) -> List[OptimizationRecommendation]:
        """Analyze posting time performance and recommend optimal times."""

        recommendations = []

        try:
            # Get time-of-day stats
            time_stats = self.performance_tracker.get_time_performance()

            if not time_stats:
                return recommendations

            # Find best and worst hours
            hours_by_performance = sorted(
                time_stats.items(),
                key=lambda x: x[1]["avg_engagement_rate"],
                reverse=True
            )

            if len(hours_by_performance) >= 3:
                # Top 3 hours
                top_hours = hours_by_performance[:3]
                bottom_hours = hours_by_performance[-3:]

                avg_top = sum(h[1]["avg_engagement_rate"] for h in top_hours) / 3
                avg_bottom = sum(h[1]["avg_engagement_rate"] for h in bottom_hours) / 3

                if avg_bottom > 0:
                    improvement = (avg_top - avg_bottom) / avg_bottom * 100

                    if improvement > 15:  # 15% better
                        recommendations.append(
                            OptimizationRecommendation(
                                category="posting_time",
                                action="replace",
                                target="posting_schedule",
                                current_value=f"Mixed hours",
                                recommended_value=f"Focus on hours: {', '.join([str(h[0]) for h in top_hours])}",
                                confidence=0.75,
                                expected_improvement=improvement,
                                reasoning=f"Top hours perform {improvement:.1f}% better on average"
                            )
                        )

        except Exception as e:
            logger.error(f"Posting time analysis failed: {e}")

        return recommendations

    def _analyze_podcast_sources(self) -> List[OptimizationRecommendation]:
        """Analyze podcast source ROI and recommend changes."""

        recommendations = []

        try:
            # Get podcast ROI
            podcast_roi = self.performance_tracker.get_podcast_roi()

            if not podcast_roi:
                return recommendations

            # Find best and worst podcasts
            podcasts_by_roi = sorted(
                podcast_roi.items(),
                key=lambda x: x[1]["avg_engagement_rate"],
                reverse=True
            )

            if len(podcasts_by_roi) >= 2:
                best = podcasts_by_roi[0]
                worst = podcasts_by_roi[-1]

                improvement = (best[1]["avg_engagement_rate"] - worst[1]["avg_engagement_rate"]) / worst[1]["avg_engagement_rate"] * 100

                if improvement > 30:  # 30% better
                    recommendations.append(
                        OptimizationRecommendation(
                            category="podcast_source",
                            action="increase",
                            target=best[0],
                            current_value=f"{best[1]['tweets_count']} tweets",
                            recommended_value="Double content from this source",
                            confidence=0.8,
                            expected_improvement=improvement / 2,
                            reasoning=f"{best[0]} has {improvement:.1f}% better engagement"
                        )
                    )

                    # Consider removing worst if very poor
                    if worst[1]["avg_engagement_rate"] < 1.0 and worst[1]["tweets_count"] >= 10:
                        recommendations.append(
                            OptimizationRecommendation(
                                category="podcast_source",
                                action="decrease",
                                target=worst[0],
                                current_value=f"{worst[1]['tweets_count']} tweets",
                                recommended_value="Reduce or pause this source",
                                confidence=0.7,
                                expected_improvement=5.0,
                                reasoning=f"{worst[0]} has very low engagement ({worst[1]['avg_engagement_rate']:.2f}%)"
                            )
                        )

        except Exception as e:
            logger.error(f"Podcast source analysis failed: {e}")

        return recommendations

    def _analyze_ab_tests(self) -> List[OptimizationRecommendation]:
        """Check A/B test results and recommend actions."""

        recommendations = []

        try:
            # Get active tests
            active_tests = self.ab_testing.get_active_tests()

            for test_id in active_tests:
                # Update metrics
                self.ab_testing.update_variant_metrics(test_id)

                # Check for winner
                winner_id = self.ab_testing.check_for_winner(test_id)

                if winner_id:
                    # Get test details
                    results = self.ab_testing.get_test_results(test_id)

                    # Find winner variant
                    winner = next((v for v in results["variants"] if v["variant_id"] == winner_id), None)

                    if winner:
                        # Calculate improvement over others
                        other_rates = [v["avg_engagement_rate"] for v in results["variants"] if v["variant_id"] != winner_id]
                        avg_other = sum(other_rates) / len(other_rates) if other_rates else 0

                        if avg_other > 0:
                            improvement = (winner["avg_engagement_rate"] - avg_other) / avg_other * 100

                            recommendations.append(
                                OptimizationRecommendation(
                                    category="ab_test_winner",
                                    action="replace",
                                    target=results["test_type"],
                                    current_value="Mixed variants",
                                    recommended_value=f"Use {winner['name']} exclusively",
                                    confidence=0.9,
                                    expected_improvement=improvement,
                                    reasoning=f"A/B test winner: {winner['name']} outperforms by {improvement:.1f}%"
                                )
                            )

                            # Declare winner
                            self.ab_testing.declare_winner(test_id, winner_id)

        except Exception as e:
            logger.error(f"A/B test analysis failed: {e}")

        return recommendations

    def auto_apply_optimizations(self,
                                 recommendations: List[OptimizationRecommendation],
                                 confidence_threshold: float = 0.8,
                                 max_changes: int = 3) -> List[OptimizationRecommendation]:
        """
        Automatically apply high-confidence optimizations.

        Args:
            recommendations: List of recommendations
            confidence_threshold: Minimum confidence to auto-apply
            max_changes: Maximum changes to apply at once

        Returns:
            List of applied recommendations
        """

        applied = []

        # Filter high-confidence recommendations
        high_confidence = [
            r for r in recommendations
            if r.confidence >= confidence_threshold
        ]

        # Sort by expected improvement
        high_confidence.sort(key=lambda x: x.expected_improvement, reverse=True)

        # Apply top N
        for rec in high_confidence[:max_changes]:
            try:
                success = self._apply_recommendation(rec)

                if success:
                    applied.append(rec)

                    # Log to history
                    self.optimization_history.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "category": rec.category,
                        "action": rec.action,
                        "target": rec.target,
                        "confidence": rec.confidence,
                        "expected_improvement": rec.expected_improvement,
                        "reasoning": rec.reasoning
                    })

            except Exception as e:
                logger.error(f"Failed to apply recommendation: {e}")

        if applied:
            self._save_optimization_history()
            logger.info(f"✅ Auto-applied {len(applied)} optimizations")

        return applied

    def _apply_recommendation(self, rec: OptimizationRecommendation) -> bool:
        """
        Apply a specific recommendation.

        Args:
            rec: Recommendation to apply

        Returns:
            True if successfully applied
        """

        # For now, just log the recommendation
        # In production, this would actually modify configuration

        logger.info(f"🔧 Applying: {rec.action} {rec.target}")
        logger.info(f"   Reasoning: {rec.reasoning}")
        logger.info(f"   Expected improvement: {rec.expected_improvement:.1f}%")

        # TODO: Implement actual configuration changes
        # This would involve:
        # - Updating viral_config.json
        # - Adjusting posting schedule
        # - Modifying hook pattern weights
        # - etc.

        return True

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""

        recommendations = self.analyze_and_recommend()

        # Get overall stats
        stats = self.performance_tracker.get_statistics()

        # Get learnings from A/B tests
        learnings = self.ab_testing.get_learnings()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_performance": stats,
            "recommendations": [
                {
                    "category": r.category,
                    "action": r.action,
                    "target": r.target,
                    "current": r.current_value,
                    "recommended": r.recommended_value,
                    "confidence": r.confidence,
                    "expected_improvement": f"{r.expected_improvement:.1f}%",
                    "reasoning": r.reasoning
                }
                for r in recommendations
            ],
            "ab_test_learnings": learnings,
            "optimization_history": self.optimization_history[-10:]  # Last 10
        }


def main():
    """CLI for feedback loop optimizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Feedback Loop Optimizer")
    parser.add_argument("--analyze", action="store_true", help="Analyze and recommend")
    parser.add_argument("--auto-apply", action="store_true", help="Auto-apply high-confidence optimizations")
    parser.add_argument("--report", action="store_true", help="Generate optimization report")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence threshold for auto-apply")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    optimizer = FeedbackLoopOptimizer()

    if args.analyze or args.auto_apply:
        recommendations = optimizer.analyze_and_recommend()

        print(f"\n📊 OPTIMIZATION RECOMMENDATIONS ({len(recommendations)}):\n")

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.category.upper()}: {rec.action} {rec.target}")
            print(f"   Current: {rec.current_value}")
            print(f"   Recommended: {rec.recommended_value}")
            print(f"   Expected improvement: {rec.expected_improvement:.1f}%")
            print(f"   Confidence: {rec.confidence:.0%}")
            print(f"   Reasoning: {rec.reasoning}")
            print()

        if args.auto_apply:
            applied = optimizer.auto_apply_optimizations(
                recommendations,
                confidence_threshold=args.confidence
            )

            print(f"\n✅ AUTO-APPLIED {len(applied)} OPTIMIZATIONS")

    elif args.report:
        report = optimizer.generate_report()
        print(json.dumps(report, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
