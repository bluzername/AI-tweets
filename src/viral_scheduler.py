"""
Viral Scheduler & Publisher - The "Engine Room" of the Podcasts TLDR machine.
Manages tweet scheduling, publishing, and provides web interface for content management.
"""

import logging
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

from .viral_tweet_crafter import ViralTweet, TweetFormat

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


class TweetQueue:
    """Manages the queue of tweets to be posted."""
    
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
                    json.dumps(scheduled_tweet.content.to_dict()),
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
            
            query += " ORDER BY scheduled_time ASC"
            
            cursor = conn.execute(query, params)
            
            tweets = []
            for row in cursor.fetchall():
                tweet_data = dict(row)
                # Reconstruct ViralTweet from JSON
                content_data = json.loads(tweet_data['content_json'])
                content_data['tweet_format'] = TweetFormat(content_data['tweet_format'])
                viral_tweet = ViralTweet(**content_data)
                
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
        self.queue = TweetQueue()
        self.timing_analyzer = OptimalTimingAnalyzer()
        
        # Scheduler state
        self._scheduler_thread = None
        self._scheduler_running = False
        
        logger.info(f"ViralScheduler initialized for {len(x_accounts)} accounts")
    
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
                # Check for ready tweets
                ready_tweets = self.queue.get_ready_tweets()
                
                for tweet in ready_tweets:
                    self._post_tweet(tweet)
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Continue after error
    
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
                wait_on_rate_limit=True
            )
            
            content = scheduled_tweet.content
            
            if content.tweet_format == TweetFormat.SINGLE:
                # Post single tweet
                response = client.create_tweet(text=content.content)
                return response.data['id'] is not None
                
            elif content.tweet_format == TweetFormat.THREAD:
                # Post thread
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