"""
Optimal Timing Analyzer - Data-driven posting time optimization.

Analyzes historical posting performance to determine the best times to post
for maximum engagement. Uses engagement metrics from performance_metrics.db
and posting history from thread_queue.db.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class TimeSlot:
    """Represents a posting time slot with engagement data."""
    hour: int
    minute: int
    day_type: str  # 'weekday' or 'weekend'
    avg_engagement: float
    sample_count: int
    confidence: float  # 0-1, based on sample size


class OptimalTimingAnalyzer:
    """
    Analyzes posting times to find optimal engagement windows.

    Uses both historical engagement data and research-based defaults
    for tech/podcast audiences (primarily US timezone).
    """

    # Research-based optimal times for tech/podcast audiences (US Eastern)
    # 8 slots per day to maximize Twitter Basic tier limit (~50 tweets/day = ~8 threads)
    DEFAULT_OPTIMAL_TIMES = {
        'weekday': [
            {'hour': 8, 'minute': 0, 'score': 0.7, 'reason': 'Morning commute (East Coast)'},
            {'hour': 9, 'minute': 30, 'score': 0.8, 'reason': 'Work start break'},
            {'hour': 10, 'minute': 30, 'score': 0.75, 'reason': 'Mid-morning break'},
            {'hour': 12, 'minute': 0, 'score': 0.9, 'reason': 'Lunch break peak'},
            {'hour': 15, 'minute': 0, 'score': 0.7, 'reason': 'Afternoon break'},
            {'hour': 17, 'minute': 30, 'score': 0.85, 'reason': 'End of workday'},
            {'hour': 20, 'minute': 0, 'score': 0.9, 'reason': 'Evening leisure peak'},
            {'hour': 21, 'minute': 30, 'score': 0.75, 'reason': 'Late evening scroll'},
        ],
        'weekend': [
            {'hour': 9, 'minute': 0, 'score': 0.7, 'reason': 'Weekend morning'},
            {'hour': 10, 'minute': 0, 'score': 0.75, 'reason': 'Late breakfast scroll'},
            {'hour': 11, 'minute': 0, 'score': 0.85, 'reason': 'Late morning leisure'},
            {'hour': 13, 'minute': 0, 'score': 0.8, 'reason': 'Early afternoon'},
            {'hour': 14, 'minute': 0, 'score': 0.9, 'reason': 'Afternoon peak'},
            {'hour': 16, 'minute': 0, 'score': 0.75, 'reason': 'Pre-dinner scroll'},
            {'hour': 17, 'minute': 0, 'score': 0.8, 'reason': 'Early evening'},
            {'hour': 20, 'minute': 0, 'score': 0.85, 'reason': 'Evening leisure'},
        ]
    }

    # Minimum samples needed to trust historical data over defaults
    MIN_SAMPLES_FOR_CONFIDENCE = 5

    def __init__(self,
                 thread_queue_db: str = "data/thread_queue.db",
                 performance_db: str = "data/performance_metrics.db"):
        """Initialize the timing analyzer."""
        self.thread_queue_db = Path(thread_queue_db)
        self.performance_db = Path(performance_db)

        # Cache for computed optimal times
        self._optimal_times_cache = None
        self._cache_timestamp = None
        self._cache_ttl = timedelta(hours=1)

    def get_historical_performance_by_hour(self) -> Dict[int, Dict]:
        """
        Analyze historical posting performance grouped by hour.

        Returns dict mapping hour -> {avg_engagement, count, weekday_count, weekend_count}
        """
        performance_by_hour = {}

        # First, get posting times from thread_queue
        if not self.thread_queue_db.exists():
            return performance_by_hour

        try:
            conn = sqlite3.connect(str(self.thread_queue_db))
            cursor = conn.cursor()

            # Get all posted threads with their times
            cursor.execute("""
                SELECT
                    thread_id,
                    posted_time,
                    strftime('%H', posted_time) as hour,
                    strftime('%w', posted_time) as day_of_week
                FROM thread_queue
                WHERE status = 'posted' AND posted_time IS NOT NULL
            """)

            posts_by_thread = {}
            for row in cursor.fetchall():
                thread_id, posted_time, hour, dow = row
                if hour and dow:
                    posts_by_thread[thread_id] = {
                        'hour': int(hour),
                        'is_weekend': int(dow) in [0, 6]  # Sunday=0, Saturday=6
                    }

            conn.close()

            # Try to get engagement data from performance_metrics
            engagement_by_thread = {}
            if self.performance_db.exists():
                try:
                    conn = sqlite3.connect(str(self.performance_db))
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT tweet_id, engagement_rate, likes, retweets, impressions
                        FROM tweet_performance
                    """)

                    for row in cursor.fetchall():
                        tweet_id, eng_rate, likes, retweets, impressions = row
                        # Use engagement rate if available, otherwise compute basic score
                        if eng_rate and eng_rate > 0:
                            engagement_by_thread[tweet_id] = eng_rate
                        elif impressions and impressions > 0:
                            engagement_by_thread[tweet_id] = ((likes or 0) + (retweets or 0) * 2) / impressions * 100

                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not read performance metrics: {e}")

            # Aggregate by hour
            for thread_id, post_data in posts_by_thread.items():
                hour = post_data['hour']
                is_weekend = post_data['is_weekend']

                if hour not in performance_by_hour:
                    performance_by_hour[hour] = {
                        'total_engagement': 0,
                        'count': 0,
                        'weekday_count': 0,
                        'weekend_count': 0,
                        'engagements': []
                    }

                # Add engagement if we have it
                engagement = engagement_by_thread.get(thread_id, None)
                if engagement is not None:
                    performance_by_hour[hour]['engagements'].append(engagement)
                    performance_by_hour[hour]['total_engagement'] += engagement

                performance_by_hour[hour]['count'] += 1
                if is_weekend:
                    performance_by_hour[hour]['weekend_count'] += 1
                else:
                    performance_by_hour[hour]['weekday_count'] += 1

            # Compute averages
            for hour, data in performance_by_hour.items():
                if data['engagements']:
                    data['avg_engagement'] = data['total_engagement'] / len(data['engagements'])
                else:
                    data['avg_engagement'] = None
                del data['total_engagement']
                del data['engagements']

            return performance_by_hour

        except Exception as e:
            logger.error(f"Error analyzing historical performance: {e}")
            return {}

    def get_optimal_times(self, force_refresh: bool = False) -> Dict[str, List[Dict]]:
        """
        Get optimal posting times, combining historical data with defaults.

        Returns dict with 'weekday' and 'weekend' keys, each containing
        a list of optimal time slots sorted by score.
        """
        # Check cache
        if not force_refresh and self._optimal_times_cache and self._cache_timestamp:
            if datetime.now() - self._cache_timestamp < self._cache_ttl:
                return self._optimal_times_cache

        historical = self.get_historical_performance_by_hour()

        # Start with defaults
        optimal = {
            'weekday': [slot.copy() for slot in self.DEFAULT_OPTIMAL_TIMES['weekday']],
            'weekend': [slot.copy() for slot in self.DEFAULT_OPTIMAL_TIMES['weekend']]
        }

        # Adjust scores based on historical data
        if historical:
            # Find the best and worst performing hours from history
            hours_with_engagement = {h: d for h, d in historical.items() if d.get('avg_engagement') is not None}

            if hours_with_engagement:
                max_engagement = max(d['avg_engagement'] for d in hours_with_engagement.values())
                min_engagement = min(d['avg_engagement'] for d in hours_with_engagement.values())
                engagement_range = max_engagement - min_engagement if max_engagement > min_engagement else 1

                # Adjust default scores based on historical performance
                for day_type in ['weekday', 'weekend']:
                    for slot in optimal[day_type]:
                        hour = slot['hour']
                        if hour in hours_with_engagement:
                            hist_data = hours_with_engagement[hour]
                            count = hist_data['count']

                            # Weight historical data based on sample size
                            confidence = min(count / self.MIN_SAMPLES_FOR_CONFIDENCE, 1.0)

                            # Normalize historical engagement to 0-1 scale
                            normalized_engagement = (hist_data['avg_engagement'] - min_engagement) / engagement_range

                            # Blend default score with historical (weighted by confidence)
                            slot['score'] = (slot['score'] * (1 - confidence * 0.5) +
                                           normalized_engagement * confidence * 0.5)
                            slot['historical_samples'] = count
                            slot['historical_engagement'] = round(hist_data['avg_engagement'], 2)

                # Sort by score
                optimal['weekday'].sort(key=lambda x: x['score'], reverse=True)
                optimal['weekend'].sort(key=lambda x: x['score'], reverse=True)

        # Add posting counts from history
        for day_type in ['weekday', 'weekend']:
            for slot in optimal[day_type]:
                hour = slot['hour']
                if hour in historical:
                    slot['posts_at_this_hour'] = historical[hour]['count']

        # Cache results
        self._optimal_times_cache = optimal
        self._cache_timestamp = datetime.now()

        return optimal

    def get_next_optimal_time(self,
                              min_hours_ahead: int = 1,
                              max_hours_ahead: int = 48,
                              avoid_hours: List[int] = None) -> datetime:
        """
        Get the next optimal posting time.

        Args:
            min_hours_ahead: Minimum hours from now
            max_hours_ahead: Maximum hours to look ahead
            avoid_hours: List of hours to avoid (e.g., if already scheduled)

        Returns:
            datetime of next optimal posting time
        """
        avoid_hours = avoid_hours or []
        optimal_times = self.get_optimal_times()

        now = datetime.now()
        earliest = now + timedelta(hours=min_hours_ahead)
        latest = now + timedelta(hours=max_hours_ahead)

        candidates = []

        # Check each day from earliest to latest
        current_date = earliest.date()
        while datetime.combine(current_date, datetime.min.time()) <= latest:
            is_weekend = current_date.weekday() >= 5
            day_type = 'weekend' if is_weekend else 'weekday'

            for slot in optimal_times[day_type]:
                candidate = datetime.combine(
                    current_date,
                    datetime.min.time().replace(hour=slot['hour'], minute=slot['minute'])
                )

                if earliest <= candidate <= latest:
                    if candidate.hour not in avoid_hours:
                        candidates.append({
                            'time': candidate,
                            'score': slot['score'],
                            'reason': slot.get('reason', 'Optimal time')
                        })

            current_date += timedelta(days=1)

        if not candidates:
            # Fallback: just use min_hours_ahead
            return earliest

        # Sort by score, then by time (prefer sooner if scores are close)
        candidates.sort(key=lambda x: (-x['score'], x['time']))

        best = candidates[0]
        logger.info(f"Next optimal time: {best['time']} (score: {best['score']:.2f}, reason: {best['reason']})")

        return best['time']

    def get_scheduled_hours(self, days_ahead: int = 2) -> List[int]:
        """
        Get hours that already have scheduled threads.

        Useful for avoiding scheduling multiple threads at the same time.
        """
        scheduled_hours = []

        if not self.thread_queue_db.exists():
            return scheduled_hours

        try:
            conn = sqlite3.connect(str(self.thread_queue_db))
            cursor = conn.cursor()

            cutoff = (datetime.now() + timedelta(days=days_ahead)).isoformat()

            cursor.execute("""
                SELECT strftime('%H', scheduled_time) as hour
                FROM thread_queue
                WHERE status = 'scheduled' AND scheduled_time <= ?
            """, (cutoff,))

            for row in cursor.fetchall():
                if row[0]:
                    scheduled_hours.append(int(row[0]))

            conn.close()

        except Exception as e:
            logger.error(f"Error getting scheduled hours: {e}")

        return scheduled_hours

    def get_timing_report(self) -> Dict:
        """
        Generate a report on posting timing optimization.

        Returns dict with analysis results for dashboard display.
        """
        historical = self.get_historical_performance_by_hour()
        optimal = self.get_optimal_times()
        scheduled = self.get_scheduled_hours()

        # Find best and worst hours from historical data
        best_hours = []
        worst_hours = []

        hours_with_data = [(h, d) for h, d in historical.items() if d.get('avg_engagement') is not None]
        if hours_with_data:
            sorted_hours = sorted(hours_with_data, key=lambda x: x[1]['avg_engagement'], reverse=True)
            best_hours = sorted_hours[:3]
            worst_hours = sorted_hours[-3:] if len(sorted_hours) >= 3 else []

        return {
            'historical_data': historical,
            'optimal_weekday': optimal['weekday'][:5],
            'optimal_weekend': optimal['weekend'][:3],
            'currently_scheduled_hours': scheduled,
            'best_performing_hours': [
                {'hour': h, 'engagement': round(d['avg_engagement'], 2), 'posts': d['count']}
                for h, d in best_hours
            ],
            'worst_performing_hours': [
                {'hour': h, 'engagement': round(d['avg_engagement'], 2), 'posts': d['count']}
                for h, d in worst_hours
            ],
            'total_posts_analyzed': sum(d['count'] for d in historical.values()),
            'recommendation': self._generate_recommendation(historical, optimal)
        }

    def _generate_recommendation(self, historical: Dict, optimal: Dict) -> str:
        """Generate a human-readable recommendation based on data."""
        total_posts = sum(d['count'] for d in historical.values())

        if total_posts < 10:
            return ("Not enough posting history yet. Using research-based optimal times "
                   "for tech/podcast audiences. Post more threads to get personalized insights.")

        # Check if we're posting at optimal times
        optimal_hours = {s['hour'] for s in optimal['weekday'][:3]} | {s['hour'] for s in optimal['weekend'][:2]}
        posts_at_optimal = sum(historical.get(h, {}).get('count', 0) for h in optimal_hours)
        optimal_ratio = posts_at_optimal / total_posts if total_posts > 0 else 0

        if optimal_ratio >= 0.6:
            return "Great job! Most of your posts are at optimal times. Keep monitoring engagement."
        elif optimal_ratio >= 0.3:
            return f"Consider posting more at optimal hours (12pm, 8pm, weekends 2pm). Currently {optimal_ratio*100:.0f}% of posts are at optimal times."
        else:
            return f"Posting timing could be improved. Only {optimal_ratio*100:.0f}% of posts are at peak engagement hours. Try 12pm and 8pm on weekdays."


# Convenience function
def get_next_optimal_posting_time(min_hours: int = 1) -> datetime:
    """Get the next optimal time to post a thread."""
    analyzer = OptimalTimingAnalyzer()
    avoid = analyzer.get_scheduled_hours()
    return analyzer.get_next_optimal_time(min_hours_ahead=min_hours, avoid_hours=avoid)


if __name__ == "__main__":
    # Test the analyzer
    logging.basicConfig(level=logging.INFO)

    analyzer = OptimalTimingAnalyzer()

    print("\n=== Timing Analysis Report ===\n")

    report = analyzer.get_timing_report()

    print(f"Total posts analyzed: {report['total_posts_analyzed']}")
    print(f"\nRecommendation: {report['recommendation']}")

    print("\n--- Best Weekday Times ---")
    for slot in report['optimal_weekday']:
        samples = slot.get('historical_samples', 0)
        print(f"  {slot['hour']:02d}:{slot['minute']:02d} - Score: {slot['score']:.2f} "
              f"({slot.get('reason', 'N/A')}) [{samples} samples]")

    print("\n--- Best Weekend Times ---")
    for slot in report['optimal_weekend']:
        samples = slot.get('historical_samples', 0)
        print(f"  {slot['hour']:02d}:{slot['minute']:02d} - Score: {slot['score']:.2f} "
              f"({slot.get('reason', 'N/A')}) [{samples} samples]")

    if report['best_performing_hours']:
        print("\n--- Best Performing Hours (Historical) ---")
        for item in report['best_performing_hours']:
            print(f"  {item['hour']:02d}:00 - {item['engagement']}% engagement ({item['posts']} posts)")

    print(f"\nNext optimal time: {analyzer.get_next_optimal_time()}")
