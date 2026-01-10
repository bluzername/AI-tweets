"""
Viral Scheduler & Publisher - The "Engine Room" of the Podcasts TLDR machine.
Manages tweet scheduling, publishing, and provides web interface for content management.
"""

import logging
import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import schedule
import time
import threading

# Scheduler check interval (default: 300 seconds = 5 minutes)
SCHEDULER_CHECK_INTERVAL = int(os.environ.get("SCHEDULER_CHECK_INTERVAL", "300"))

from .viral_tweet_crafter import ViralTweet, TweetFormat
from .optimal_timing import OptimalTimingAnalyzer as DataDrivenTimingAnalyzer, get_next_optimal_posting_time
from .x_rate_manager import XRateLimitManager, XRateLimitConfig

logger = logging.getLogger(__name__)


class TweetStatus(Enum):
    """Status of tweets in the queue."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    POSTED = "posted"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PostingStrategy(Enum):
    """Different posting strategies for viral optimization."""
    IMMEDIATE = "immediate"
    OPTIMAL_TIME = "optimal_time"
    A_B_TEST = "a_b_test"
    DRIP_CAMPAIGN = "drip_campaign"


@dataclass
class ScheduledTweet:
    """Represents a tweet in the scheduling queue."""
    
    # Core content
    tweet_id: str
    content: ViralTweet
    account_name: str
    
    # Scheduling
    scheduled_time: datetime
    status: TweetStatus = TweetStatus.DRAFT
    posting_strategy: PostingStrategy = PostingStrategy.OPTIMAL_TIME
    
    # Source tracking
    episode_id: Optional[str] = None
    insight_id: Optional[str] = None
    podcast_name: Optional[str] = None
    
    # Performance tracking
    posted_time: Optional[datetime] = None
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0
    
    # A/B testing
    variant_group: Optional[str] = None
    is_control_group: bool = False
    
    # Metadata
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['posting_strategy'] = self.posting_strategy.value
        data['content'] = self.content.to_dict()
        # Convert datetime objects to ISO strings
        for field in ['scheduled_time', 'posted_time', 'created_at', 'updated_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data


class OptimalTimingAnalyzer:
    """Analyzes optimal posting times based on historical data."""
    
    def __init__(self, db_path: str = "data/scheduling.db"):
        """Initialize timing analyzer."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Default optimal times (can be refined with data)
        self.default_optimal_times = {
            'weekday': [
                {'hour': 9, 'minute': 0, 'engagement_score': 0.8},   # Morning commute
                {'hour': 12, 'minute': 30, 'engagement_score': 0.9}, # Lunch break
                {'hour': 15, 'minute': 0, 'engagement_score': 0.7},  # Afternoon break
                {'hour': 18, 'minute': 0, 'engagement_score': 0.8},  # End of workday
                {'hour': 20, 'minute': 30, 'engagement_score': 0.9}  # Evening leisure
            ],
            'weekend': [
                {'hour': 10, 'minute': 0, 'engagement_score': 0.8},  # Weekend morning
                {'hour': 14, 'minute': 0, 'engagement_score': 0.9},  # Weekend afternoon
                {'hour': 19, 'minute': 0, 'engagement_score': 0.8}   # Weekend evening
            ]
        }
    
    def get_next_optimal_time(self, 
                            account_name: str,
                            content_type: str = "single",
                            min_hours_ahead: int = 1) -> datetime:
        """Get the next optimal posting time for an account."""
        
        now = datetime.now()
        base_time = now + timedelta(hours=min_hours_ahead)
        
        # Determine if it's weekend or weekday
        weekday = base_time.weekday()
        is_weekend = weekday >= 5
        
        optimal_times = self.default_optimal_times['weekend' if is_weekend else 'weekday']
        
        # Find next optimal time slot
        for time_slot in optimal_times:
            candidate_time = base_time.replace(
                hour=time_slot['hour'],
                minute=time_slot['minute'],
                second=0,
                microsecond=0
            )
            
            # If today's slot has passed, try tomorrow
            if candidate_time <= now:
                candidate_time += timedelta(days=1)
                # Recheck if it's still the right day type
                if candidate_time.weekday() >= 5 != is_weekend:
                    continue
            
            # Check if slot is available (not too close to other scheduled tweets)
            if self._is_time_slot_available(account_name, candidate_time):
                return candidate_time
        
        # Fallback: just add hours to current time
        return base_time + timedelta(hours=2)
    
    def _is_time_slot_available(self, 
                              account_name: str, 
                              target_time: datetime,
                              min_gap_minutes: int = 30) -> bool:
        """Check if a time slot is available (no tweets within min_gap_minutes)."""
        # This would check the database for nearby scheduled tweets
        # For now, return True (simplified)
        return True
    
    def update_engagement_data(self, 
                             account_name: str,
                             posted_time: datetime,
                             engagement_metrics: Dict[str, int]):
        """Update engagement data to improve timing predictions."""
        # This would store engagement data and update optimal timing models
        # Implementation would analyze which times perform best
        pass


class ThreadQueue:
    """Manages the queue of threads to be posted (new thread-based storage)."""

    def __init__(self, db_path: str = "data/thread_queue.db"):
        """Initialize thread queue database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create thread queue table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_queue (
                    thread_id TEXT PRIMARY KEY,
                    account_name TEXT NOT NULL,
                    podcast_name TEXT,
                    podcast_handle TEXT,
                    host_handles TEXT,
                    guest_handles TEXT,
                    episode_id TEXT,
                    episode_title TEXT,
                    tweets_json TEXT NOT NULL,
                    thumbnail_path TEXT,
                    scheduled_time TEXT NOT NULL,
                    status TEXT DEFAULT 'scheduled',
                    posted_time TEXT,
                    first_tweet_id TEXT,
                    all_tweet_ids TEXT,
                    hook_variant TEXT,
                    hook_type TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add all_tweet_ids column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE thread_queue ADD COLUMN all_tweet_ids TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Add hook_variant column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE thread_queue ADD COLUMN hook_variant TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Add hook_type column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE thread_queue ADD COLUMN hook_type TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Add platform column if it doesn't exist (migration for multi-platform support)
            try:
                conn.execute("ALTER TABLE thread_queue ADD COLUMN platform TEXT DEFAULT 'x'")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_account_status ON thread_queue (account_name, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_scheduled_time ON thread_queue (scheduled_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_episode_id ON thread_queue (episode_id)")

    def add_thread(self,
                   thread_tweets: List[str],
                   account_name: str,
                   podcast_name: str = None,
                   podcast_handle: str = None,
                   host_handles: List[str] = None,
                   guest_handles: List[str] = None,
                   episode_id: str = None,
                   episode_title: str = None,
                   thumbnail_path: str = None,
                   scheduled_time: datetime = None,
                   use_optimal_timing: bool = True,
                   hook_variant: str = None,
                   hook_type: str = None,
                   platform: str = 'x') -> Optional[str]:
        """
        Add a complete thread to the queue.

        Args:
            thread_tweets: List of tweet strings in order (6 tweets typically)
            account_name: X.com account to post from
            podcast_name: Name of the podcast
            podcast_handle: @handle of the podcast
            host_handles: List of host @handles
            guest_handles: List of guest @handles
            episode_id: Source episode ID
            episode_title: Title of the episode
            thumbnail_path: Local path to thumbnail image
            scheduled_time: When to post the thread
            use_optimal_timing: If True and scheduled_time is None, use data-driven optimal timing
            hook_variant: The specific hook text used (for A/B testing)
            hook_type: Type of hook used (question, stat, controversy, etc.)
            platform: Target platform - 'x', 'telegram', or 'both'

        Returns:
            thread_id if successful, None otherwise
        """
        try:
            thread_id = f"{account_name}_{episode_id}_{int(datetime.now().timestamp())}"

            if scheduled_time is None:
                if use_optimal_timing:
                    # Use data-driven optimal timing analyzer
                    scheduled_time = get_next_optimal_posting_time(min_hours=1)
                    logger.info(f"Using optimal timing: {scheduled_time}")
                else:
                    scheduled_time = datetime.now() + timedelta(hours=1)

            # Auto-detect hook type from first tweet if not provided
            if hook_type is None and thread_tweets:
                hook_type = self._detect_hook_type(thread_tweets[0])

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO thread_queue
                    (thread_id, account_name, podcast_name, podcast_handle,
                     host_handles, guest_handles, episode_id, episode_title,
                     tweets_json, thumbnail_path, scheduled_time, status,
                     hook_variant, hook_type, platform)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    thread_id,
                    account_name,
                    podcast_name,
                    podcast_handle,
                    json.dumps(host_handles or [], ensure_ascii=False),
                    json.dumps(guest_handles or [], ensure_ascii=False),
                    episode_id,
                    episode_title,
                    json.dumps(thread_tweets, ensure_ascii=False),
                    thumbnail_path,
                    scheduled_time.isoformat(),
                    'scheduled',
                    hook_variant,
                    hook_type,
                    platform
                ))

            logger.info(f"Added thread to queue: {thread_id} ({len(thread_tweets)} tweets, hook: {hook_type})")
            return thread_id

        except Exception as e:
            logger.error(f"Error adding thread to queue: {e}")
            return None

    def _detect_hook_type(self, first_tweet: str) -> str:
        """
        Auto-detect the type of hook used in the first tweet.

        Args:
            first_tweet: The opening tweet of the thread

        Returns:
            Hook type: question, stat, controversy, story, insight, quote
        """
        # Check for question patterns
        if '?' in first_tweet:
            return 'question'

        # Check for statistic/number patterns
        if any(char.isdigit() for char in first_tweet) and any(w in first_tweet.lower() for w in ['%', 'million', 'billion', 'people', 'years']):
            return 'stat'

        # Check for controversial/surprising patterns
        controversial_words = ['never', 'always', 'wrong', 'myth', 'actually', 'truth', 'secret', 'unpopular']
        if any(word in first_tweet.lower() for word in controversial_words):
            return 'controversy'

        # Check for quote patterns (starts with quote mark)
        if first_tweet.strip().startswith('"') or first_tweet.strip().startswith('"'):
            return 'quote'

        # Check for story patterns
        story_words = ['i was', 'i learned', 'years ago', 'my experience', 'journey', 'story']
        if any(word in first_tweet.lower() for word in story_words):
            return 'story'

        # Default to insight
        return 'insight'

    def get_ready_threads(self, account_name: str = None, platform: str = None) -> List[Dict]:
        """Get threads ready for posting.
        
        Args:
            account_name: Filter by account name
            platform: Filter by platform ('x', 'telegram', or None for all)
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT * FROM thread_queue
                WHERE status = 'scheduled'
                AND scheduled_time <= ?
            """
            params = [now]

            if account_name:
                query += " AND account_name = ?"
                params.append(account_name)
            
            if platform:
                query += " AND (platform = ? OR platform = 'both')"
                params.append(platform)

            query += " ORDER BY scheduled_time ASC LIMIT 1"  # Only 1 thread per cycle

            cursor = conn.execute(query, params)
            threads = []

            for row in cursor.fetchall():
                thread_data = dict(row)
                # Parse JSON fields
                thread_data['tweets'] = json.loads(thread_data['tweets_json'])
                thread_data['host_handles'] = json.loads(thread_data['host_handles']) if thread_data['host_handles'] else []
                thread_data['guest_handles'] = json.loads(thread_data['guest_handles']) if thread_data['guest_handles'] else []
                thread_data['all_tweet_ids'] = json.loads(thread_data['all_tweet_ids']) if thread_data.get('all_tweet_ids') else []
                threads.append(thread_data)

            return threads

    def get_all_threads(self, account_name: str = None, status: str = None, limit: int = 50) -> List[Dict]:
        """Get all threads with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM thread_queue WHERE 1=1"
            params = []

            if account_name:
                query += " AND account_name = ?"
                params.append(account_name)

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY scheduled_time DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            threads = []

            for row in cursor.fetchall():
                thread_data = dict(row)
                thread_data['tweets'] = json.loads(thread_data['tweets_json'])
                thread_data['host_handles'] = json.loads(thread_data['host_handles']) if thread_data['host_handles'] else []
                thread_data['guest_handles'] = json.loads(thread_data['guest_handles']) if thread_data['guest_handles'] else []
                thread_data['all_tweet_ids'] = json.loads(thread_data['all_tweet_ids']) if thread_data.get('all_tweet_ids') else []
                threads.append(thread_data)

            return threads

    def update_thread_status(self,
                            thread_id: str,
                            status: str,
                            **kwargs):
        """Update thread status and metadata."""
        updates = {"status": status, "updated_at": datetime.now().isoformat()}
        updates.update(kwargs)

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE thread_queue SET {set_clause} WHERE thread_id = ?",
                (*updates.values(), thread_id)
            )

    def get_thread_stats(self, account_name: str = None) -> Dict[str, int]:
        """Get statistics about the thread queue."""
        with sqlite3.connect(self.db_path) as conn:
            query_base = "SELECT status, COUNT(*) FROM thread_queue"
            params = []

            if account_name:
                query_base += " WHERE account_name = ?"
                params.append(account_name)

            query_base += " GROUP BY status"

            cursor = conn.execute(query_base, params)
            stats = dict(cursor.fetchall())

            return {
                'scheduled': stats.get('scheduled', 0),
                'posted': stats.get('posted', 0),
                'failed': stats.get('failed', 0),
                'total': sum(stats.values())
            }


class TweetQueue:
    """Manages the queue of tweets to be posted (legacy - kept for backwards compatibility)."""

    def __init__(self, db_path: str = "data/tweet_queue.db"):
        """Initialize tweet queue database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create tweet queue tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tweet_queue (
                    tweet_id TEXT PRIMARY KEY,
                    account_name TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    scheduled_time TEXT NOT NULL,
                    status TEXT DEFAULT 'draft',
                    posting_strategy TEXT DEFAULT 'optimal_time',
                    episode_id TEXT,
                    insight_id TEXT,
                    podcast_name TEXT,
                    posted_time TEXT,
                    likes INTEGER DEFAULT 0,
                    retweets INTEGER DEFAULT 0,
                    replies INTEGER DEFAULT 0,
                    impressions INTEGER DEFAULT 0,
                    engagement_rate REAL DEFAULT 0.0,
                    variant_group TEXT,
                    is_control_group BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_account_status ON tweet_queue (account_name, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scheduled_time ON tweet_queue (scheduled_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_id ON tweet_queue (episode_id)")
    
    def add_tweet(self, scheduled_tweet: ScheduledTweet) -> bool:
        """Add a tweet to the queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tweet_queue 
                    (tweet_id, account_name, content_json, scheduled_time, status,
                     posting_strategy, episode_id, insight_id, podcast_name,
                     variant_group, is_control_group, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    scheduled_tweet.tweet_id,
                    scheduled_tweet.account_name,
                    json.dumps(scheduled_tweet.content.to_dict(), ensure_ascii=False),
                    scheduled_tweet.scheduled_time.isoformat(),
                    scheduled_tweet.status.value,
                    scheduled_tweet.posting_strategy.value,
                    scheduled_tweet.episode_id,
                    scheduled_tweet.insight_id,
                    scheduled_tweet.podcast_name,
                    scheduled_tweet.variant_group,
                    scheduled_tweet.is_control_group,
                    scheduled_tweet.created_at.isoformat(),
                    scheduled_tweet.updated_at.isoformat()
                ))
            
            logger.info(f"Added tweet to queue: {scheduled_tweet.tweet_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding tweet to queue: {e}")
            return False
    
    def get_ready_tweets(self, account_name: str = None) -> List[ScheduledTweet]:
        """Get tweets ready for posting."""
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = """
                SELECT * FROM tweet_queue 
                WHERE status = 'scheduled' 
                AND scheduled_time <= ?
            """
            params = [now]
            
            if account_name:
                query += " AND account_name = ?"
                params.append(account_name)
            
            query += " ORDER BY scheduled_time ASC LIMIT 1"  # Only 1 thread per cycle
            
            cursor = conn.execute(query, params)
            
            tweets = []
            for row in cursor.fetchall():
                tweet_data = dict(row)
                # Reconstruct ViralTweet from JSON
                content_data = json.loads(tweet_data['content_json'])
                content_data['tweet_format'] = TweetFormat(content_data['tweet_format'])

                # Filter to only valid ViralTweet fields to avoid unknown keyword arguments
                valid_fields = {
                    'content', 'tweet_format', 'character_count', 'thread_tweets', 'thread_count',
                    'poll_question', 'poll_options', 'quote_card_text', 'quote_card_author',
                    'hashtags', 'mentions', 'timestamp_link', 'hooks', 'cta', 'engagement_triggers',
                    'predicted_engagement', 'viral_score', 'target_audience'
                }
                filtered_content = {k: v for k, v in content_data.items() if k in valid_fields}
                viral_tweet = ViralTweet(**filtered_content)
                
                # Create ScheduledTweet
                scheduled_tweet = ScheduledTweet(
                    tweet_id=tweet_data['tweet_id'],
                    content=viral_tweet,
                    account_name=tweet_data['account_name'],
                    scheduled_time=datetime.fromisoformat(tweet_data['scheduled_time']),
                    status=TweetStatus(tweet_data['status']),
                    posting_strategy=PostingStrategy(tweet_data['posting_strategy']),
                    episode_id=tweet_data['episode_id'],
                    insight_id=tweet_data['insight_id'],
                    podcast_name=tweet_data['podcast_name'],
                    variant_group=tweet_data['variant_group'],
                    is_control_group=bool(tweet_data['is_control_group']),
                    created_at=datetime.fromisoformat(tweet_data['created_at']),
                    updated_at=datetime.fromisoformat(tweet_data['updated_at'])
                )
                
                tweets.append(scheduled_tweet)
            
            return tweets
    
    def update_tweet_status(self, 
                          tweet_id: str, 
                          status: TweetStatus,
                          **kwargs):
        """Update tweet status and metadata."""
        updates = {"status": status.value, "updated_at": datetime.now().isoformat()}
        updates.update(kwargs)
        
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE tweet_queue SET {set_clause} WHERE tweet_id = ?",
                (*updates.values(), tweet_id)
            )
    
    def get_queue_stats(self, account_name: str = None) -> Dict[str, int]:
        """Get statistics about the tweet queue."""
        with sqlite3.connect(self.db_path) as conn:
            query_base = "SELECT status, COUNT(*) FROM tweet_queue"
            params = []
            
            if account_name:
                query_base += " WHERE account_name = ?"
                params.append(account_name)
            
            query_base += " GROUP BY status"
            
            cursor = conn.execute(query_base, params)
            stats = dict(cursor.fetchall())
            
            return {
                'draft': stats.get('draft', 0),
                'scheduled': stats.get('scheduled', 0),
                'posted': stats.get('posted', 0),
                'failed': stats.get('failed', 0),
                'cancelled': stats.get('cancelled', 0),
                'total': sum(stats.values())
            }


class ViralScheduler:
    """Main scheduler that orchestrates tweet posting and optimization."""

    def __init__(self,
                 x_accounts: Dict[str, Dict],
                 posting_frequency: Dict[str, int] = None):
        """
        Initialize viral scheduler.

        Args:
            x_accounts: Dict of account_name -> X.com credentials
            posting_frequency: Dict of account_name -> tweets_per_day
        """
        self.accounts = x_accounts
        self.posting_frequency = posting_frequency or {}

        # Initialize components
        self.queue = TweetQueue()  # Legacy individual tweet queue
        self.thread_queue = ThreadQueue()  # New thread-based queue
        self.timing_analyzer = OptimalTimingAnalyzer()
        
        # Rate limit manager for robust posting
        self.rate_manager = XRateLimitManager(XRateLimitConfig(
            tweets_per_month=1500,
            tweets_per_15min_window=15,
            min_seconds_between_tweets=5.0,
            min_seconds_between_threads=180.0
        ))

        # Scheduler state
        self._scheduler_thread = None
        self._scheduler_running = False

        # Tweepy API instances for media upload (requires v1.1 API)
        self._api_instances = {}

        logger.info(f"ViralScheduler initialized for {len(x_accounts)} accounts")

    def _get_api_instance(self, account_name: str):
        """Get Tweepy API v1.1 instance for media uploads."""
        if account_name in self._api_instances:
            return self._api_instances[account_name]

        account_config = self.accounts.get(account_name)
        if not account_config:
            return None

        try:
            import tweepy

            auth = tweepy.OAuth1UserHandler(
                account_config.get('consumer_key'),
                account_config.get('consumer_secret'),
                account_config.get('access_token'),
                account_config.get('access_token_secret')
            )
            api = tweepy.API(auth)
            self._api_instances[account_name] = api
            return api
        except Exception as e:
            logger.error(f"Error creating API instance for {account_name}: {e}")
            return None

    def _get_client_instance(self, account_config: Dict):
        """Get Tweepy Client v2 instance for posting tweets."""
        import tweepy

        return tweepy.Client(
            bearer_token=account_config.get('bearer_token'),
            consumer_key=account_config.get('consumer_key'),
            consumer_secret=account_config.get('consumer_secret'),
            access_token=account_config.get('access_token'),
            access_token_secret=account_config.get('access_token_secret'),
            wait_on_rate_limit=False
        )

    def schedule_thread(self,
                        thread_tweets: List[str],
                        account_name: str,
                        podcast_name: str = None,
                        podcast_handle: str = None,
                        host_handles: List[str] = None,
                        guest_handles: List[str] = None,
                        episode_id: str = None,
                        episode_title: str = None,
                        thumbnail_path: str = None,
                        scheduled_time: datetime = None,
                        platforms: List[str] = None) -> Optional[str]:
        """
        Schedule a complete thread for posting to one or more platforms.

        Args:
            thread_tweets: List of tweet strings in order
            account_name: X.com account to post from
            podcast_name: Name of the podcast
            podcast_handle: @handle of the podcast
            host_handles: List of host @handles
            guest_handles: List of guest @handles
            episode_id: Source episode ID
            episode_title: Title of the episode
            thumbnail_path: Local path to thumbnail image
            scheduled_time: When to post the thread
            platforms: List of platforms to post to ['x', 'telegram'] - defaults to ['x', 'telegram']

        Returns:
            thread_id if successful, None otherwise
        """
        # Default to both platforms
        if platforms is None:
            platforms = ['x', 'telegram']
        
        thread_id = None
        
        for platform in platforms:
            tid = self.thread_queue.add_thread(
                thread_tweets=thread_tweets,
                account_name=account_name,
                podcast_name=podcast_name,
                podcast_handle=podcast_handle,
                host_handles=host_handles,
                guest_handles=guest_handles,
                episode_id=episode_id,
                episode_title=episode_title,
                thumbnail_path=thumbnail_path,
                scheduled_time=scheduled_time,
                platform=platform
            )
            if tid:
                thread_id = tid  # Return the last successful ID
                logger.info(f"Scheduled thread for {platform}: {tid}")
        
        return thread_id

    def _post_thread(self, thread_record: Dict) -> bool:
        """
        Post an entire thread as connected replies with media.
        Uses XRateLimitManager for robust rate limit handling.

        Args:
            thread_record: Dict containing thread data from database

        Returns:
            True if all tweets posted successfully, False otherwise
        """
        try:
            account_name = thread_record['account_name']
            account_config = self.accounts.get(account_name)

            if not account_config:
                logger.error(f"No config found for account: {account_name}")
                self.thread_queue.update_thread_status(thread_record['thread_id'], 'failed')
                return False

            tweets = thread_record['tweets']
            thumbnail_path = thread_record.get('thumbnail_path')
            
            # Check rate limits before attempting to post
            can_post, reason, wait_time = self.rate_manager.can_post_thread(len(tweets))
            
            if not can_post:
                if wait_time > 0 and wait_time <= 300:  # Wait up to 5 minutes
                    logger.info(f"Rate limit: waiting {wait_time:.0f}s - {reason}")
                    time.sleep(wait_time)
                    can_post, reason, wait_time = self.rate_manager.can_post_thread(len(tweets))
                
                if not can_post:
                    logger.warning(f"Cannot post thread (rate limit): {reason}")
                    # Add to retry queue instead of failing immediately
                    self.rate_manager.add_to_retry_queue(thread_record, 0)
                    return False

            # Get API instances
            client = self._get_client_instance(account_config)
            api = self._get_api_instance(account_name)

            tweet_ids = []
            logger.info(f"Posting thread {thread_record['thread_id']} ({len(tweets)} tweets)")
            
            # Log rate limit status
            status = self.rate_manager.get_status()
            logger.info(f"Rate limit status: {status['tweets_in_window']}/{status['window_limit']} in window, "
                       f"{status['monthly_count']}/{status['monthly_limit']} monthly")

            for i, tweet_text in enumerate(tweets):
                try:
                    # Wait for rate limit clearance
                    self.rate_manager.wait_for_tweet()
                    
                    if i == 0:
                        # First tweet: include thumbnail if available
                        if thumbnail_path and Path(thumbnail_path).exists() and api:
                            logger.info(f"Uploading media: {thumbnail_path}")
                            media = api.media_upload(thumbnail_path)
                            response = client.create_tweet(
                                text=tweet_text,
                                media_ids=[media.media_id]
                            )
                        else:
                            response = client.create_tweet(text=tweet_text)
                    else:
                        # Reply to previous tweet
                        response = client.create_tweet(
                            text=tweet_text,
                            in_reply_to_tweet_id=tweet_ids[-1]
                        )

                    tweet_ids.append(response.data['id'])
                    self.rate_manager.record_tweet()
                    logger.info(f"Posted tweet {i+1}/{len(tweets)}: {response.data['id']}")

                except Exception as e:
                    error_str = str(e).lower()
                    if 'rate limit' in error_str or '429' in error_str:
                        logger.warning(f"Rate limit hit at tweet {i+1}, adding to retry queue")
                        self.rate_manager.add_to_retry_queue(thread_record, 0)
                        # Partial success - update what we have
                        if tweet_ids:
                            self.thread_queue.update_thread_status(
                                thread_record['thread_id'],
                                'failed',
                                first_tweet_id=tweet_ids[0],
                                all_tweet_ids=json.dumps(tweet_ids),
                                posted_time=datetime.now().isoformat()
                            )
                        return False
                    
                    logger.error(f"Error posting tweet {i+1} in thread: {e}")
                    self.thread_queue.update_thread_status(
                        thread_record['thread_id'],
                        'failed',
                        first_tweet_id=tweet_ids[0] if tweet_ids else None,
                        all_tweet_ids=json.dumps(tweet_ids) if tweet_ids else None,
                        posted_time=datetime.now().isoformat()
                    )
                    return False

            # All tweets posted successfully
            self.rate_manager.record_thread_complete()
            self.thread_queue.update_thread_status(
                thread_record['thread_id'],
                'posted',
                first_tweet_id=tweet_ids[0] if tweet_ids else None,
                all_tweet_ids=json.dumps(tweet_ids) if tweet_ids else None,
                posted_time=datetime.now().isoformat()
            )

            logger.info(f"Successfully posted thread {thread_record['thread_id']} - {len(tweet_ids)} tweets")
            return True

        except Exception as e:
            logger.error(f"Error posting thread {thread_record.get('thread_id')}: {e}")
            self.thread_queue.update_thread_status(thread_record['thread_id'], 'failed')
            return False

    def schedule_tweets(self, 
                       tweets: List[ViralTweet],
                       account_name: str,
                       episode_id: str = None,
                       podcast_name: str = None,
                       strategy: PostingStrategy = PostingStrategy.OPTIMAL_TIME) -> List[str]:
        """
        Schedule a batch of tweets for posting.
        
        Args:
            tweets: List of viral tweets to schedule
            account_name: Target X.com account
            episode_id: Source episode ID
            podcast_name: Source podcast name
            strategy: Posting strategy to use
            
        Returns:
            List of tweet IDs that were scheduled
        """
        logger.info(f"Scheduling {len(tweets)} tweets for {account_name}")
        
        scheduled_ids = []
        
        for i, tweet in enumerate(tweets):
            # Generate unique tweet ID
            tweet_id = f"{account_name}_{episode_id}_{i}_{int(datetime.now().timestamp())}"
            
            # Determine scheduling time based on strategy
            if strategy == PostingStrategy.IMMEDIATE:
                scheduled_time = datetime.now() + timedelta(minutes=1)
            elif strategy == PostingStrategy.OPTIMAL_TIME:
                scheduled_time = self.timing_analyzer.get_next_optimal_time(
                    account_name, 
                    tweet.tweet_format.value,
                    min_hours_ahead=1 + i  # Space out tweets
                )
            elif strategy == PostingStrategy.DRIP_CAMPAIGN:
                # Space tweets over several days
                scheduled_time = datetime.now() + timedelta(days=i, hours=12)
            else:
                scheduled_time = datetime.now() + timedelta(hours=1 + i)
            
            # Create scheduled tweet
            scheduled_tweet = ScheduledTweet(
                tweet_id=tweet_id,
                content=tweet,
                account_name=account_name,
                scheduled_time=scheduled_time,
                status=TweetStatus.SCHEDULED,
                posting_strategy=strategy,
                episode_id=episode_id,
                podcast_name=podcast_name
            )
            
            # Add to queue
            if self.queue.add_tweet(scheduled_tweet):
                scheduled_ids.append(tweet_id)
        
        logger.info(f"Scheduled {len(scheduled_ids)} tweets for {account_name}")
        return scheduled_ids
    
    def start_scheduler(self):
        """Start the automated scheduler thread."""
        if self._scheduler_running:
            logger.warning("Scheduler already running")
            return
        
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=self._run_scheduler)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
        
        logger.info("Viral scheduler started")
    
    def stop_scheduler(self):
        """Stop the automated scheduler."""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join()
        logger.info("Viral scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                # Check for ready threads (new thread-based queue) - ONE at a time
                ready_threads = self.thread_queue.get_ready_threads()
                if ready_threads:
                    thread = ready_threads[0]  # Only process first thread
                    logger.info(f"Processing thread: {thread['thread_id']}")
                    success = self._post_thread(thread)
                    
                    if success:
                        # Wait 20 minutes after successful post before next attempt
                        logger.info("Thread posted successfully. Waiting 20 minutes before next post.")
                        time.sleep(1200)  # 20 minutes
                    else:
                        # Wait 30 minutes after failed post (likely rate limited)
                        logger.info("Thread failed (rate limit). Waiting 30 minutes before retry.")
                        time.sleep(1800)  # 30 minutes

                # Check for ready individual tweets (legacy queue)
                ready_tweets = self.queue.get_ready_tweets()
                for tweet in ready_tweets:
                    self._post_tweet(tweet)

                # Sleep before next check (default: 5 minutes)
                time.sleep(SCHEDULER_CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(SCHEDULER_CHECK_INTERVAL)  # Continue after error
    
    def _post_tweet(self, scheduled_tweet: ScheduledTweet) -> bool:
        """Post a single tweet."""
        try:
            logger.info(f"Posting tweet: {scheduled_tweet.tweet_id}")
            
            account_config = self.accounts.get(scheduled_tweet.account_name)
            if not account_config:
                logger.error(f"No config found for account: {scheduled_tweet.account_name}")
                self.queue.update_tweet_status(scheduled_tweet.tweet_id, TweetStatus.FAILED)
                return False
            
            # Post the tweet using X.com API
            success = self._post_to_x(scheduled_tweet, account_config)
            
            if success:
                self.queue.update_tweet_status(
                    scheduled_tweet.tweet_id, 
                    TweetStatus.POSTED,
                    posted_time=datetime.now().isoformat()
                )
                logger.info(f"Successfully posted tweet: {scheduled_tweet.tweet_id}")
                return True
            else:
                self.queue.update_tweet_status(scheduled_tweet.tweet_id, TweetStatus.FAILED)
                return False
                
        except Exception as e:
            logger.error(f"Error posting tweet {scheduled_tweet.tweet_id}: {e}")
            self.queue.update_tweet_status(scheduled_tweet.tweet_id, TweetStatus.FAILED)
            return False
    
    def _post_to_x(self, scheduled_tweet: ScheduledTweet, account_config: Dict) -> bool:
        """Post tweet to X.com using the API."""
        try:
            import tweepy
            
            # Initialize X.com client
            client = tweepy.Client(
                bearer_token=account_config.get('bearer_token'),
                consumer_key=account_config.get('consumer_key'),
                consumer_secret=account_config.get('consumer_secret'),
                access_token=account_config.get('access_token'),
                access_token_secret=account_config.get('access_token_secret'),
                wait_on_rate_limit=False
            )
            
            content = scheduled_tweet.content

            if content.tweet_format == TweetFormat.SINGLE:
                # Post single tweet
                response = client.create_tweet(text=content.content)
                return response.data['id'] is not None

            elif content.tweet_format == TweetFormat.THREAD:
                # Check if we have thread_tweets array or just content
                if content.thread_tweets and len(content.thread_tweets) > 0:
                    # Post full thread
                    tweet_ids = []
                    for i, tweet_text in enumerate(content.thread_tweets):
                        if i == 0:
                            # First tweet
                            response = client.create_tweet(text=tweet_text)
                            tweet_ids.append(response.data['id'])
                        else:
                            # Reply to previous tweet
                            response = client.create_tweet(
                                text=tweet_text,
                                in_reply_to_tweet_id=tweet_ids[-1]
                            )
                            tweet_ids.append(response.data['id'])

                    return len(tweet_ids) == len(content.thread_tweets)
                else:
                    # Single tweet within a thread (stored as individual rows)
                    # Just post the content as a single tweet
                    response = client.create_tweet(text=content.content)
                    return response.data['id'] is not None
                
            elif content.tweet_format == TweetFormat.POLL:
                # Post poll
                response = client.create_tweet(
                    text=content.content,
                    poll_options=content.poll_options,
                    poll_duration_minutes=1440  # 24 hours
                )
                return response.data['id'] is not None
            
            else:
                # Quote card or other formats - post as regular tweet for now
                response = client.create_tweet(text=content.content)
                return response.data['id'] is not None
                
        except Exception as e:
            logger.error(f"Error posting to X.com: {e}")
            return False
    
    def get_performance_stats(self, 
                            account_name: str = None,
                            days_back: int = 7) -> Dict[str, Any]:
        """Get performance statistics for posted tweets."""
        # This would query the database for engagement metrics
        # and provide analytics on what's working
        
        stats = self.queue.get_queue_stats(account_name)
        
        # Add performance metrics
        stats.update({
            'avg_engagement_rate': 0.0,  # Would calculate from real data
            'best_posting_time': "12:30 PM",  # Would analyze from data
            'top_performing_format': "thread",  # Would analyze from data
            'total_impressions': 0,  # Would sum from real data
            'viral_tweets_count': 0  # Would count high-performing tweets
        })
        
        return stats