#!/usr/bin/env python3
"""
Cost Tracker - Monitor and optimize API costs.
Tracks spending across OpenAI, Anthropic, and Google APIs.
"""

import logging
import sqlite3
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class APICost:
    """Cost for a single API call."""
    service: str  # "openai", "anthropic", "google"
    operation: str  # "transcription", "completion", "embedding"
    model: str
    tokens_used: Optional[int]
    cost_usd: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class CostTracker:
    """
    Track and analyze API costs.

    Features:
    - Per-service cost tracking
    - Daily/monthly budgets
    - Cost optimization recommendations
    - Usage forecasting
    - Alert on budget thresholds
    """

    # Cost per 1K tokens (approximate, update with real prices)
    COST_RATES = {
        "openai": {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "whisper-1": 0.006,  # per minute
            "text-embedding-3-small": 0.00002,
        },
        "anthropic": {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        },
        "google": {
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
        }
    }

    def __init__(self, db_path: str = "data/cost_tracking.db"):
        """Initialize cost tracker."""
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize cost tracking database."""

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL,
                operation TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_used INTEGER,
                cost_usd REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON api_costs(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_service
            ON api_costs(service)
        """)

        conn.commit()
        conn.close()

        logger.info(f"✅ Cost tracking database initialized: {self.db_path}")

    def track_cost(self, cost: APICost):
        """Record an API cost."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO api_costs
            (service, operation, model, tokens_used, cost_usd, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            cost.service,
            cost.operation,
            cost.model,
            cost.tokens_used,
            cost.cost_usd,
            cost.timestamp,
            json.dumps(cost.metadata) if cost.metadata else None
        ))

        conn.commit()
        conn.close()

        logger.debug(f"💰 Tracked ${cost.cost_usd:.4f} - {cost.service}/{cost.operation}")

    def get_daily_cost(self, date: Optional[str] = None) -> float:
        """Get total cost for a specific day."""

        if date is None:
            date = datetime.utcnow().date().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT SUM(cost_usd)
            FROM api_costs
            WHERE date(timestamp) = date(?)
        """, (date,))

        result = cursor.fetchone()
        conn.close()

        return result[0] or 0.0

    def get_monthly_cost(self, year_month: Optional[str] = None) -> float:
        """Get total cost for a specific month (YYYY-MM)."""

        if year_month is None:
            year_month = datetime.utcnow().strftime("%Y-%m")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT SUM(cost_usd)
            FROM api_costs
            WHERE strftime('%Y-%m', timestamp) = ?
        """, (year_month,))

        result = cursor.fetchone()
        conn.close()

        return result[0] or 0.0

    def get_cost_breakdown(self, days: int = 7) -> Dict[str, Dict[str, float]]:
        """Get cost breakdown by service and operation."""

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT service, operation, SUM(cost_usd), COUNT(*)
            FROM api_costs
            WHERE timestamp >= ?
            GROUP BY service, operation
            ORDER BY SUM(cost_usd) DESC
        """, (cutoff,))

        breakdown = {}

        for service, operation, total_cost, count in cursor.fetchall():
            if service not in breakdown:
                breakdown[service] = {}

            breakdown[service][operation] = {
                "cost": round(total_cost, 4),
                "count": count,
                "avg_cost": round(total_cost / count, 6)
            }

        conn.close()

        return breakdown

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations."""

        recommendations = []

        # Get recent breakdown
        breakdown = self.get_cost_breakdown(days=7)

        # Check for high-cost operations
        for service, operations in breakdown.items():
            for operation, stats in operations.items():
                # High cost operations (>$1/call on average)
                if stats["avg_cost"] > 1.0:
                    recommendations.append({
                        "type": "high_cost_operation",
                        "service": service,
                        "operation": operation,
                        "avg_cost": stats["avg_cost"],
                        "recommendation": f"Consider caching or using cheaper alternatives for {operation}",
                        "potential_savings": stats["cost"] * 0.5  # Assume 50% reduction possible
                    })

                # High volume low-value operations
                if stats["count"] > 100 and stats["avg_cost"] < 0.01:
                    recommendations.append({
                        "type": "high_volume_operation",
                        "service": service,
                        "operation": operation,
                        "count": stats["count"],
                        "recommendation": f"High volume detected for {operation} - consider batching",
                        "potential_savings": stats["cost"] * 0.2
                    })

        # Check for model optimization opportunities
        if "openai" in breakdown:
            if "completion" in breakdown["openai"]:
                comp_cost = breakdown["openai"]["completion"]["cost"]

                if comp_cost > 5.0:  # Significant completion costs
                    recommendations.append({
                        "type": "model_optimization",
                        "service": "openai",
                        "operation": "completion",
                        "cost": comp_cost,
                        "recommendation": "Consider using GPT-3.5-turbo instead of GPT-4 for simple tasks",
                        "potential_savings": comp_cost * 0.7  # GPT-3.5 is ~30% of GPT-4 cost
                    })

        return recommendations

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cost statistics."""

        today_cost = self.get_daily_cost()
        month_cost = self.get_monthly_cost()
        breakdown = self.get_cost_breakdown(days=30)

        # Total cost
        total_all_time = 0.0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT SUM(cost_usd) FROM api_costs")
        result = cursor.fetchone()
        if result[0]:
            total_all_time = result[0]

        conn.close()

        return {
            "today_cost": round(today_cost, 2),
            "month_cost": round(month_cost, 2),
            "total_all_time": round(total_all_time, 2),
            "breakdown_30d": breakdown,
            "avg_daily_cost_30d": round(month_cost / 30, 2) if month_cost else 0
        }

    def check_budget(self, daily_limit: float, monthly_limit: float) -> Dict[str, Any]:
        """Check if within budget limits."""

        today_cost = self.get_daily_cost()
        month_cost = self.get_monthly_cost()

        return {
            "daily": {
                "cost": round(today_cost, 2),
                "limit": daily_limit,
                "percentage": round((today_cost / daily_limit * 100), 1) if daily_limit > 0 else 0,
                "exceeded": today_cost > daily_limit
            },
            "monthly": {
                "cost": round(month_cost, 2),
                "limit": monthly_limit,
                "percentage": round((month_cost / monthly_limit * 100), 1) if monthly_limit > 0 else 0,
                "exceeded": month_cost > monthly_limit
            }
        }


def main():
    """CLI for cost tracking."""
    import argparse

    parser = argparse.ArgumentParser(description="API Cost Tracker")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--breakdown", action="store_true", help="Show cost breakdown")
    parser.add_argument("--recommendations", action="store_true", help="Get optimization recommendations")
    parser.add_argument("--budget", action="store_true", help="Check budget status")
    parser.add_argument("--daily-limit", type=float, default=10.0, help="Daily budget limit")
    parser.add_argument("--monthly-limit", type=float, default=300.0, help="Monthly budget limit")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tracker = CostTracker()

    if args.stats:
        stats = tracker.get_statistics()
        print(json.dumps(stats, indent=2))

    elif args.breakdown:
        breakdown = tracker.get_cost_breakdown(days=30)
        print("\n💰 COST BREAKDOWN (Last 30 days):\n")

        for service, operations in breakdown.items():
            print(f"\n{service.upper()}:")
            for operation, stats in operations.items():
                print(f"  {operation}:")
                print(f"    Total: ${stats['cost']:.4f}")
                print(f"    Calls: {stats['count']}")
                print(f"    Avg: ${stats['avg_cost']:.6f}/call")

    elif args.recommendations:
        recs = tracker.get_optimization_recommendations()
        print(f"\n💡 COST OPTIMIZATION RECOMMENDATIONS ({len(recs)}):\n")

        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['type'].upper()}")
            print(f"   {rec['recommendation']}")
            print(f"   Potential savings: ${rec['potential_savings']:.2f}")
            print()

    elif args.budget:
        budget_status = tracker.check_budget(args.daily_limit, args.monthly_limit)
        print(json.dumps(budget_status, indent=2))

        if budget_status["daily"]["exceeded"]:
            print("\n⚠️ DAILY BUDGET EXCEEDED!")

        if budget_status["monthly"]["exceeded"]:
            print("\n⚠️ MONTHLY BUDGET EXCEEDED!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
