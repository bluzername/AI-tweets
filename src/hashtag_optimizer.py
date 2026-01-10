"""
Hashtag Optimizer - Strategic hashtag research and optimization for podcast threads.

Manages curated hashtags for tech/podcast audiences, tracks performance,
and provides recommendations based on content analysis.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class HashtagRecommendation:
    """A hashtag recommendation with metadata."""
    hashtag: str
    category: str
    relevance_score: float  # 0-1
    avg_engagement: Optional[float] = None
    usage_count: int = 0
    last_used: Optional[str] = None


class HashtagOptimizer:
    """
    Strategic hashtag optimization for podcast threads.

    Features:
    - Curated hashtag database for tech/podcast audiences
    - Content-based hashtag suggestions
    - Performance tracking (which hashtags drive engagement)
    - Rotation to avoid shadowbanning from overuse
    """

    # Curated hashtags for podcast/tech audiences
    CURATED_HASHTAGS = {
        # Primary podcast hashtags
        'podcast': {
            'hashtags': ['podcast', 'podcasts', 'podcasting', 'podcastlife', 'podcastshow'],
            'priority': 'high',
            'max_per_thread': 1
        },
        # Tech industry hashtags
        'tech': {
            'hashtags': ['tech', 'technology', 'innovation', 'AI', 'startup', 'startups'],
            'priority': 'high',
            'max_per_thread': 1
        },
        # Business/entrepreneurship
        'business': {
            'hashtags': ['business', 'entrepreneur', 'entrepreneurship', 'leadership', 'success'],
            'priority': 'medium',
            'max_per_thread': 1
        },
        # Personal development
        'growth': {
            'hashtags': ['personaldevelopment', 'selfimprovement', 'motivation', 'mindset', 'growth'],
            'priority': 'medium',
            'max_per_thread': 1
        },
        # Content format
        'content': {
            'hashtags': ['threadworthy', 'mustread', 'takeaways', 'insights', 'lessons'],
            'priority': 'low',
            'max_per_thread': 1
        },
        # Trending tech topics
        'trending_tech': {
            'hashtags': ['ChatGPT', 'OpenAI', 'LLM', 'crypto', 'Web3', 'SaaS'],
            'priority': 'conditional',
            'max_per_thread': 1
        },
        # Finance/money
        'finance': {
            'hashtags': ['money', 'finance', 'investing', 'wealth', 'financialfreedom'],
            'priority': 'medium',
            'max_per_thread': 1
        },
        # Health/wellness
        'wellness': {
            'hashtags': ['health', 'wellness', 'mentalhealth', 'mindfulness', 'productivity'],
            'priority': 'medium',
            'max_per_thread': 1
        }
    }

    # Keyword to category mapping for content analysis
    KEYWORD_CATEGORIES = {
        'ai': ['tech', 'trending_tech'],
        'artificial intelligence': ['tech', 'trending_tech'],
        'machine learning': ['tech', 'trending_tech'],
        'startup': ['tech', 'business'],
        'entrepreneur': ['business'],
        'business': ['business'],
        'money': ['finance', 'business'],
        'invest': ['finance'],
        'wealth': ['finance'],
        'health': ['wellness'],
        'mental': ['wellness'],
        'productiv': ['growth', 'wellness'],
        'success': ['growth', 'business'],
        'mindset': ['growth'],
        'leadership': ['business'],
        'innovation': ['tech'],
        'crypto': ['trending_tech', 'finance'],
        'bitcoin': ['trending_tech', 'finance'],
        'chatgpt': ['trending_tech', 'tech'],
        'openai': ['trending_tech', 'tech'],
        'saas': ['tech', 'business'],
    }

    def __init__(self,
                 db_path: str = "data/hashtag_metrics.db",
                 max_hashtags_per_tweet: int = 3,
                 rotation_days: int = 7):
        """
        Initialize hashtag optimizer.

        Args:
            db_path: Path to hashtag metrics database
            max_hashtags_per_tweet: Max hashtags to include per tweet
            rotation_days: Days before reusing same hashtag heavily
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.max_hashtags = max_hashtags_per_tweet
        self.rotation_days = rotation_days
        self._init_database()

    def _init_database(self):
        """Create hashtag tracking tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Track hashtag usage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hashtag_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hashtag TEXT NOT NULL,
                    thread_id TEXT,
                    tweet_id TEXT,
                    used_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    category TEXT
                )
            """)

            # Track hashtag performance (when we have engagement data)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hashtag_performance (
                    hashtag TEXT PRIMARY KEY,
                    total_uses INTEGER DEFAULT 0,
                    total_impressions INTEGER DEFAULT 0,
                    total_engagements INTEGER DEFAULT 0,
                    avg_engagement_rate REAL DEFAULT 0,
                    last_used TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_hashtag ON hashtag_usage (hashtag)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_thread ON hashtag_usage (thread_id)")

    def get_hashtags_for_content(self,
                                  content: str,
                                  podcast_name: str = None,
                                  episode_title: str = None,
                                  max_hashtags: int = None) -> List[str]:
        """
        Get optimal hashtags for content.

        Args:
            content: The text content to analyze
            podcast_name: Name of the podcast (for context)
            episode_title: Title of the episode (for context)
            max_hashtags: Override default max hashtags

        Returns:
            List of hashtags (without # prefix)
        """
        max_tags = max_hashtags or self.max_hashtags

        # Combine all text for analysis
        full_text = f"{content} {podcast_name or ''} {episode_title or ''}".lower()

        # Detect relevant categories based on content
        relevant_categories = self._detect_categories(full_text)

        # Get recent usage to implement rotation
        recent_usage = self._get_recent_usage()

        # Build hashtag candidates with scores
        candidates = []

        # Always include one podcast hashtag
        candidates.append(HashtagRecommendation(
            hashtag='podcast',
            category='podcast',
            relevance_score=0.9
        ))

        # Add category-based hashtags
        for category in relevant_categories:
            if category in self.CURATED_HASHTAGS:
                cat_data = self.CURATED_HASHTAGS[category]
                for tag in cat_data['hashtags']:
                    # Calculate score based on priority and recent usage
                    base_score = {'high': 0.8, 'medium': 0.6, 'low': 0.4, 'conditional': 0.5}.get(
                        cat_data['priority'], 0.5
                    )

                    # Reduce score if used recently
                    usage_penalty = recent_usage.get(tag, 0) * 0.1
                    final_score = max(0, base_score - usage_penalty)

                    candidates.append(HashtagRecommendation(
                        hashtag=tag,
                        category=category,
                        relevance_score=final_score,
                        usage_count=recent_usage.get(tag, 0)
                    ))

        # Sort by relevance score and deduplicate
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        selected = []
        seen_categories = set()

        for candidate in candidates:
            if len(selected) >= max_tags:
                break

            # Only one hashtag per category to avoid spam
            if candidate.category in seen_categories:
                continue

            # Skip if duplicate hashtag
            if candidate.hashtag in [s.hashtag for s in selected]:
                continue

            selected.append(candidate)
            seen_categories.add(candidate.category)

        return [f"#{h.hashtag}" for h in selected]

    def _detect_categories(self, text: str) -> List[str]:
        """Detect content categories from text."""
        categories = set()

        for keyword, cats in self.KEYWORD_CATEGORIES.items():
            if keyword in text:
                categories.update(cats)

        # Always include podcast category for our content
        categories.add('podcast')

        return list(categories)

    def _get_recent_usage(self) -> Dict[str, int]:
        """Get hashtag usage count from recent days."""
        usage = {}
        cutoff = (datetime.now() - timedelta(days=self.rotation_days)).isoformat()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT hashtag, COUNT(*) as count
                    FROM hashtag_usage
                    WHERE used_at >= ?
                    GROUP BY hashtag
                """, (cutoff,))

                for row in cursor.fetchall():
                    usage[row[0]] = row[1]
        except Exception as e:
            logger.warning(f"Could not get recent usage: {e}")

        return usage

    def record_usage(self,
                     hashtags: List[str],
                     thread_id: str = None,
                     tweet_id: str = None):
        """Record hashtag usage for tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for tag in hashtags:
                    # Remove # prefix if present
                    clean_tag = tag.lstrip('#')

                    # Find category
                    category = 'unknown'
                    for cat, data in self.CURATED_HASHTAGS.items():
                        if clean_tag in data['hashtags']:
                            category = cat
                            break

                    conn.execute("""
                        INSERT INTO hashtag_usage (hashtag, thread_id, tweet_id, category)
                        VALUES (?, ?, ?, ?)
                    """, (clean_tag, thread_id, tweet_id, category))

                    # Update performance tracking
                    conn.execute("""
                        INSERT INTO hashtag_performance (hashtag, total_uses, last_used, updated_at)
                        VALUES (?, 1, ?, ?)
                        ON CONFLICT(hashtag) DO UPDATE SET
                            total_uses = total_uses + 1,
                            last_used = excluded.last_used,
                            updated_at = excluded.updated_at
                    """, (clean_tag, datetime.now().isoformat(), datetime.now().isoformat()))

        except Exception as e:
            logger.error(f"Error recording hashtag usage: {e}")

    def update_performance(self,
                           hashtag: str,
                           impressions: int = 0,
                           engagements: int = 0):
        """Update hashtag performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current stats
                cursor = conn.execute("""
                    SELECT total_impressions, total_engagements, total_uses
                    FROM hashtag_performance WHERE hashtag = ?
                """, (hashtag.lstrip('#'),))
                row = cursor.fetchone()

                if row:
                    new_impressions = row[0] + impressions
                    new_engagements = row[1] + engagements
                    total_uses = row[2]

                    avg_rate = (new_engagements / new_impressions * 100) if new_impressions > 0 else 0

                    conn.execute("""
                        UPDATE hashtag_performance
                        SET total_impressions = ?, total_engagements = ?,
                            avg_engagement_rate = ?, updated_at = ?
                        WHERE hashtag = ?
                    """, (new_impressions, new_engagements, avg_rate,
                          datetime.now().isoformat(), hashtag.lstrip('#')))

        except Exception as e:
            logger.error(f"Error updating hashtag performance: {e}")

    def get_performance_report(self) -> Dict:
        """Get hashtag performance report."""
        report = {
            'top_performing': [],
            'most_used': [],
            'recommendations': []
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Top performing by engagement rate
                cursor = conn.execute("""
                    SELECT hashtag, avg_engagement_rate, total_uses, total_impressions
                    FROM hashtag_performance
                    WHERE total_uses >= 3
                    ORDER BY avg_engagement_rate DESC
                    LIMIT 10
                """)
                report['top_performing'] = [
                    {'hashtag': row[0], 'engagement_rate': round(row[1], 2),
                     'uses': row[2], 'impressions': row[3]}
                    for row in cursor.fetchall()
                ]

                # Most used
                cursor = conn.execute("""
                    SELECT hashtag, total_uses, last_used
                    FROM hashtag_performance
                    ORDER BY total_uses DESC
                    LIMIT 10
                """)
                report['most_used'] = [
                    {'hashtag': row[0], 'uses': row[1], 'last_used': row[2]}
                    for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.error(f"Error getting performance report: {e}")

        return report

    def add_hashtags_to_tweet(self,
                               tweet_text: str,
                               hashtags: List[str],
                               position: str = 'end') -> str:
        """
        Add hashtags to tweet text.

        Args:
            tweet_text: Original tweet text
            hashtags: List of hashtags to add
            position: 'end' or 'inline'

        Returns:
            Tweet text with hashtags added
        """
        if not hashtags:
            return tweet_text

        # Ensure hashtags have # prefix
        formatted_tags = [f"#{h.lstrip('#')}" for h in hashtags]
        hashtag_str = ' '.join(formatted_tags)

        # Check character limit
        combined = f"{tweet_text}\n\n{hashtag_str}"
        if len(combined) <= 280:
            return combined

        # Try without double newline
        combined = f"{tweet_text} {hashtag_str}"
        if len(combined) <= 280:
            return combined

        # Reduce hashtags if needed
        while hashtags and len(f"{tweet_text} {' '.join(formatted_tags)}") > 280:
            hashtags = hashtags[:-1]
            formatted_tags = [f"#{h.lstrip('#')}" for h in hashtags]

        if hashtags:
            return f"{tweet_text} {' '.join(formatted_tags)}"

        return tweet_text


def get_hashtags_for_thread(content: str,
                            podcast_name: str = None,
                            max_tags: int = 3) -> List[str]:
    """Convenience function to get hashtags for a thread."""
    optimizer = HashtagOptimizer()
    return optimizer.get_hashtags_for_content(content, podcast_name, max_hashtags=max_tags)


if __name__ == "__main__":
    # Test the optimizer
    logging.basicConfig(level=logging.INFO)

    optimizer = HashtagOptimizer()

    # Test content
    test_content = """
    The key to building a successful startup is understanding that failure
    is part of the journey. AI and machine learning are transforming how
    we build products. Focus on mental health and productivity to maintain
    momentum.
    """

    hashtags = optimizer.get_hashtags_for_content(
        test_content,
        podcast_name="The Tim Ferriss Show",
        episode_title="How to Build a Billion Dollar Company"
    )

    print("\n=== Hashtag Optimizer Test ===\n")
    print(f"Content sample: {test_content[:100]}...")
    print(f"\nRecommended hashtags: {hashtags}")

    # Test adding to tweet
    sample_tweet = "This is a sample tweet about startup success and AI innovation."
    with_hashtags = optimizer.add_hashtags_to_tweet(sample_tweet, hashtags)
    print(f"\nTweet with hashtags ({len(with_hashtags)} chars):")
    print(with_hashtags)

    # Record usage
    optimizer.record_usage(hashtags, thread_id="test_thread_1")

    # Get report
    report = optimizer.get_performance_report()
    print(f"\nPerformance report: {json.dumps(report, indent=2)}")
