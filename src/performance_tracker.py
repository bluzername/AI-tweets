#!/usr/bin/env python3
"""
Performance Tracker - Automated engagement metrics collection and analysis.
Tracks tweet performance and learns what works for continuous improvement.
"""

import logging
import os
import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class TweetPerformance:
    """Performance metrics for a tweet."""
    tweet_id: str
    content: str
    posted_at: str

    # Engagement metrics
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    quotes: int = 0
    bookmarks: int = 0
    impressions: int = 0

    # Derived metrics
    engagement_rate: float = 0.0  # (likes + RTs + replies) / impressions
    viral_score: float = 0.0  # Weighted engagement score

    # Context
    hook_pattern: Optional[str] = None
    hook_variant_id: Optional[str] = None
    podcast_name: Optional[str] = None
    episode_id: Optional[str] = None
    category: Optional[str] = None
    post_hour: Optional[int] = None  # Hour of day posted (0-23)
    post_day: Optional[str] = None  # Day of week

    # Tracking
    last_updated: Optional[str] = None
    collection_count: int = 0  # How many times we've collected metrics


class PerformanceTracker:
    """
    Track and analyze tweet performance metrics.

    Features:
    - Automated metrics collection from Twitter API
    - Performance database
    - Engagement rate calculation
    - Hook pattern performance analysis
    - Time-of-day performance analysis
    - Podcast ROI calculation
    - Trend detection
    """

    def __init__(self,
                 db_path: str = "data/performance_metrics.db",
                 twitter_credentials: Optional[Dict[str, str]] = None):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to metrics database
            twitter_credentials: Twitter API credentials
        """

        self.db_path = db_path
        self.twitter_credentials = twitter_credentials or {}

        # Twitter API client
        self.twitter_client = None
        self._init_twitter_client()

        # Initialize database
        self._init_database()

    def _init_twitter_client(self):
        """Initialize Twitter API client."""

        # Try to get credentials from environment if not provided
        if not self.twitter_credentials:
            self.twitter_credentials = {
                "consumer_key": os.getenv("MAIN_API_KEY", ""),
                "consumer_secret": os.getenv("MAIN_API_SECRET", ""),
                "access_token": os.getenv("MAIN_ACCESS_TOKEN", ""),
                "access_token_secret": os.getenv("MAIN_ACCESS_TOKEN_SECRET", ""),
                "bearer_token": os.getenv("MAIN_BEARER_TOKEN", "")
            }

        # Check if we have credentials
        if not self.twitter_credentials.get("bearer_token"):
            logger.warning("⚠️ Twitter credentials not configured - metrics collection disabled")
            return

        try:
            import tweepy

            # Initialize Tweepy client
            self.twitter_client = tweepy.Client(
                bearer_token=self.twitter_credentials["bearer_token"],
                consumer_key=self.twitter_credentials["consumer_key"],
                consumer_secret=self.twitter_credentials["consumer_secret"],
                access_token=self.twitter_credentials["access_token"],
                access_token_secret=self.twitter_credentials["access_token_secret"],
                wait_on_rate_limit=True
            )

            logger.info("✅ Twitter API client initialized")

        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Twitter client: {e}")

    def _init_database(self):
        """Initialize performance metrics database."""

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tweet_performance (
                tweet_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                posted_at TEXT NOT NULL,

                -- Engagement metrics
                likes INTEGER DEFAULT 0,
                retweets INTEGER DEFAULT 0,
                replies INTEGER DEFAULT 0,
                quotes INTEGER DEFAULT 0,
                bookmarks INTEGER DEFAULT 0,
                impressions INTEGER DEFAULT 0,

                -- Derived metrics
                engagement_rate REAL DEFAULT 0.0,
                viral_score REAL DEFAULT 0.0,

                -- Context
                hook_pattern TEXT,
                hook_variant_id TEXT,
                podcast_name TEXT,
                episode_id TEXT,
                category TEXT,
                post_hour INTEGER,
                post_day TEXT,

                -- Tracking
                last_updated TEXT,
                collection_count INTEGER DEFAULT 0,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for analysis
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_posted_at
            ON tweet_performance(posted_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hook_pattern
            ON tweet_performance(hook_pattern)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_podcast_name
            ON tweet_performance(podcast_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_post_hour
            ON tweet_performance(post_hour)
        """)

        conn.commit()
        conn.close()

        logger.info(f"✅ Performance database initialized: {self.db_path}")

    def collect_metrics(self, tweet_id: str) -> Optional[TweetPerformance]:
        """
        Collect performance metrics for a tweet from Twitter API.

        Args:
            tweet_id: Twitter tweet ID

        Returns:
            TweetPerformance object or None if collection failed
        """

        if not self.twitter_client:
            logger.warning("⚠️ Twitter client not initialized, cannot collect metrics")
            return None

        try:
            # Get tweet with metrics
            tweet = self.twitter_client.get_tweet(
                id=tweet_id,
                tweet_fields=["public_metrics", "created_at", "text"],
                user_auth=True
            )

            if not tweet.data:
                logger.warning(f"⚠️ Tweet not found: {tweet_id}")
                return None

            tweet_data = tweet.data
            metrics = tweet_data.public_metrics

            # Parse posted time
            posted_at = tweet_data.created_at.isoformat()
            post_hour = tweet_data.created_at.hour
            post_day = tweet_data.created_at.strftime("%A")

            # Calculate engagement rate
            total_engagement = metrics["like_count"] + metrics["retweet_count"] + metrics["reply_count"]
            impressions = metrics.get("impression_count", 0)
            engagement_rate = (total_engagement / impressions * 100) if impressions > 0 else 0.0

            # Calculate viral score (weighted)
            viral_score = (
                metrics["like_count"] * 1.0 +
                metrics["retweet_count"] * 3.0 +  # RTs worth more
                metrics["reply_count"] * 2.0 +
                metrics.get("quote_tweet_count", 0) * 2.5
            ) / 100.0  # Normalize

            performance = TweetPerformance(
                tweet_id=tweet_id,
                content=tweet_data.text,
                posted_at=posted_at,
                likes=metrics["like_count"],
                retweets=metrics["retweet_count"],
                replies=metrics["reply_count"],
                quotes=metrics.get("quote_tweet_count", 0),
                bookmarks=metrics.get("bookmark_count", 0),
                impressions=impressions,
                engagement_rate=engagement_rate,
                viral_score=viral_score,
                post_hour=post_hour,
                post_day=post_day,
                last_updated=datetime.utcnow().isoformat()
            )

            logger.info(f"✅ Metrics collected for {tweet_id}: {total_engagement} engagements, {engagement_rate:.2f}% rate")

            return performance

        except Exception as e:
            logger.error(f"❌ Failed to collect metrics for {tweet_id}: {e}")
            return None

    def save_performance(self, performance: TweetPerformance):
        """Save or update performance metrics in database."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if exists
        cursor.execute("SELECT collection_count FROM tweet_performance WHERE tweet_id = ?", (performance.tweet_id,))
        existing = cursor.fetchone()

        if existing:
            # Update
            collection_count = existing[0] + 1

            cursor.execute("""
                UPDATE tweet_performance
                SET likes = ?, retweets = ?, replies = ?, quotes = ?, bookmarks = ?,
                    impressions = ?, engagement_rate = ?, viral_score = ?,
                    last_updated = ?, collection_count = ?
                WHERE tweet_id = ?
            """, (
                performance.likes, performance.retweets, performance.replies,
                performance.quotes, performance.bookmarks, performance.impressions,
                performance.engagement_rate, performance.viral_score,
                performance.last_updated, collection_count,
                performance.tweet_id
            ))

        else:
            # Insert
            cursor.execute("""
                INSERT INTO tweet_performance
                (tweet_id, content, posted_at, likes, retweets, replies, quotes, bookmarks,
                 impressions, engagement_rate, viral_score, hook_pattern, hook_variant_id,
                 podcast_name, episode_id, category, post_hour, post_day, last_updated, collection_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance.tweet_id, performance.content, performance.posted_at,
                performance.likes, performance.retweets, performance.replies,
                performance.quotes, performance.bookmarks, performance.impressions,
                performance.engagement_rate, performance.viral_score,
                performance.hook_pattern, performance.hook_variant_id,
                performance.podcast_name, performance.episode_id, performance.category,
                performance.post_hour, performance.post_day,
                performance.last_updated, 1
            ))

        conn.commit()
        conn.close()

        logger.info(f"✅ Performance saved for {performance.tweet_id}")

    def collect_recent_metrics(self, hours: int = 24):
        """
        Collect metrics for all tweets posted in the last N hours.

        Args:
            hours: Look back this many hours
        """

        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get tweets that need metric collection
        cursor.execute("""
            SELECT tweet_id
            FROM tweet_performance
            WHERE posted_at >= ?
            AND (last_updated IS NULL OR last_updated < datetime('now', '-1 hour'))
        """, (cutoff,))

        tweet_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not tweet_ids:
            logger.info("No tweets need metric collection")
            return

        logger.info(f"📊 Collecting metrics for {len(tweet_ids)} tweets...")

        collected = 0
        failed = 0

        for tweet_id in tweet_ids:
            performance = self.collect_metrics(tweet_id)

            if performance:
                self.save_performance(performance)
                collected += 1
            else:
                failed += 1

            # Rate limiting - wait between calls
            time.sleep(1)

        logger.info(f"✅ Collection complete: {collected} succeeded, {failed} failed")

    def get_hook_performance(self, min_samples: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by hook pattern.

        Args:
            min_samples: Minimum samples required for reliable stats

        Returns:
            Dictionary of hook pattern stats
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT hook_pattern,
                   COUNT(*) as count,
                   AVG(engagement_rate) as avg_engagement_rate,
                   AVG(viral_score) as avg_viral_score,
                   AVG(likes) as avg_likes,
                   AVG(retweets) as avg_retweets
            FROM tweet_performance
            WHERE hook_pattern IS NOT NULL
            AND impressions > 0
            GROUP BY hook_pattern
            HAVING count >= ?
            ORDER BY avg_engagement_rate DESC
        """, (min_samples,))

        results = {}

        for row in cursor.fetchall():
            pattern, count, eng_rate, viral, likes, rts = row

            results[pattern] = {
                "count": count,
                "avg_engagement_rate": round(eng_rate, 2),
                "avg_viral_score": round(viral, 2),
                "avg_likes": round(likes, 1),
                "avg_retweets": round(rts, 1)
            }

        conn.close()

        return results

    def get_time_performance(self) -> Dict[int, Dict[str, float]]:
        """
        Analyze performance by time of day.

        Returns:
            Dictionary of hourly performance stats
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT post_hour,
                   COUNT(*) as count,
                   AVG(engagement_rate) as avg_engagement_rate,
                   AVG(viral_score) as avg_viral_score
            FROM tweet_performance
            WHERE post_hour IS NOT NULL
            AND impressions > 0
            GROUP BY post_hour
            ORDER BY post_hour
        """)

        results = {}

        for row in cursor.fetchall():
            hour, count, eng_rate, viral = row

            results[hour] = {
                "count": count,
                "avg_engagement_rate": round(eng_rate, 2),
                "avg_viral_score": round(viral, 2)
            }

        conn.close()

        return results

    def get_podcast_roi(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate ROI per podcast source.

        Returns:
            Dictionary of podcast performance
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT podcast_name,
                   COUNT(*) as tweets_count,
                   AVG(engagement_rate) as avg_engagement_rate,
                   AVG(viral_score) as avg_viral_score,
                   SUM(likes + retweets + replies) as total_engagement
            FROM tweet_performance
            WHERE podcast_name IS NOT NULL
            AND impressions > 0
            GROUP BY podcast_name
            ORDER BY avg_engagement_rate DESC
        """)

        results = {}

        for row in cursor.fetchall():
            podcast, count, eng_rate, viral, total_eng = row

            results[podcast] = {
                "tweets_count": count,
                "avg_engagement_rate": round(eng_rate, 2),
                "avg_viral_score": round(viral, 2),
                "total_engagement": total_eng,
                "roi_score": round(eng_rate * count / 10, 2)  # Weighted score
            }

        conn.close()

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total tweets tracked
        cursor.execute("SELECT COUNT(*) FROM tweet_performance")
        total_tweets = cursor.fetchone()[0]

        # Average metrics
        cursor.execute("""
            SELECT AVG(engagement_rate), AVG(viral_score), AVG(likes), AVG(retweets)
            FROM tweet_performance
            WHERE impressions > 0
        """)
        avg_stats = cursor.fetchone()

        # Best performing tweet
        cursor.execute("""
            SELECT tweet_id, content, engagement_rate, viral_score
            FROM tweet_performance
            WHERE impressions > 0
            ORDER BY viral_score DESC
            LIMIT 1
        """)
        best_tweet = cursor.fetchone()

        conn.close()

        return {
            "total_tweets": total_tweets,
            "avg_engagement_rate": round(avg_stats[0], 2) if avg_stats[0] else 0,
            "avg_viral_score": round(avg_stats[1], 2) if avg_stats[1] else 0,
            "avg_likes": round(avg_stats[2], 1) if avg_stats[2] else 0,
            "avg_retweets": round(avg_stats[3], 1) if avg_stats[3] else 0,
            "best_tweet": {
                "id": best_tweet[0],
                "content": best_tweet[1][:100] + "..." if len(best_tweet[1]) > 100 else best_tweet[1],
                "engagement_rate": round(best_tweet[2], 2),
                "viral_score": round(best_tweet[3], 2)
            } if best_tweet else None
        }


def main():
    """CLI for performance tracking."""
    import argparse

    parser = argparse.ArgumentParser(description="Tweet Performance Tracker")
    parser.add_argument("--collect", type=str, help="Collect metrics for tweet ID")
    parser.add_argument("--collect-recent", type=int, help="Collect for recent N hours")
    parser.add_argument("--hook-stats", action="store_true", help="Show hook performance")
    parser.add_argument("--time-stats", action="store_true", help="Show time-of-day performance")
    parser.add_argument("--podcast-roi", action="store_true", help="Show podcast ROI")
    parser.add_argument("--stats", action="store_true", help="Show overall statistics")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tracker = PerformanceTracker()

    if args.collect:
        performance = tracker.collect_metrics(args.collect)
        if performance:
            tracker.save_performance(performance)
            print(json.dumps(asdict(performance), indent=2))

    elif args.collect_recent:
        tracker.collect_recent_metrics(hours=args.collect_recent)

    elif args.hook_stats:
        stats = tracker.get_hook_performance()
        print("\n📊 HOOK PATTERN PERFORMANCE:\n")
        for pattern, metrics in sorted(stats.items(), key=lambda x: x[1]["avg_engagement_rate"], reverse=True):
            print(f"{pattern}:")
            print(f"  Tweets: {metrics['count']}")
            print(f"  Avg Engagement Rate: {metrics['avg_engagement_rate']}%")
            print(f"  Avg Viral Score: {metrics['avg_viral_score']}")
            print()

    elif args.time_stats:
        stats = tracker.get_time_performance()
        print("\n⏰ TIME-OF-DAY PERFORMANCE:\n")
        for hour in sorted(stats.keys()):
            metrics = stats[hour]
            print(f"{hour:02d}:00 - Engagement: {metrics['avg_engagement_rate']:.2f}% ({metrics['count']} tweets)")

    elif args.podcast_roi:
        roi = tracker.get_podcast_roi()
        print("\n💰 PODCAST ROI:\n")
        for podcast, metrics in sorted(roi.items(), key=lambda x: x[1]["roi_score"], reverse=True):
            print(f"{podcast}:")
            print(f"  Tweets: {metrics['tweets_count']}")
            print(f"  Avg Engagement: {metrics['avg_engagement_rate']}%")
            print(f"  ROI Score: {metrics['roi_score']}")
            print()

    elif args.stats:
        stats = tracker.get_statistics()
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
