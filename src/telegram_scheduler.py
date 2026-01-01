#!/usr/bin/env python3
"""
Telegram Scheduler - Handles Telegram-specific posting logic.
Much simpler than X scheduler since there are no rate limits.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from .telegram_publisher import TelegramPublisher
from .viral_scheduler import ThreadQueue

logger = logging.getLogger(__name__)


class TelegramScheduler:
    """
    Scheduler for Telegram channel posts.
    
    Unlike X scheduler, this can post immediately without rate limit concerns.
    """
    
    def __init__(self, telegram_config: Dict[str, Dict], thread_queue: ThreadQueue = None):
        """
        Initialize Telegram scheduler.
        
        Args:
            telegram_config: Dict of channel_name -> {bot_token, channel_id, enabled}
            thread_queue: ThreadQueue instance (shared with X scheduler)
        """
        self.config = telegram_config
        self.thread_queue = thread_queue or ThreadQueue()
        self.publishers: Dict[str, TelegramPublisher] = {}
        
        # Initialize publishers for enabled channels
        for channel_name, channel_config in telegram_config.items():
            if channel_config.get('enabled', False):
                self.publishers[channel_name] = TelegramPublisher(
                    bot_token=channel_config['bot_token'],
                    channel_id=channel_config['channel_id']
                )
                logger.info(f"Initialized Telegram publisher: {channel_name}")
        
        logger.info(f"TelegramScheduler initialized with {len(self.publishers)} channels")
    
    def get_channel_for_account(self, x_account_name: str) -> Optional[str]:
        """
        Map X account name to Telegram channel name.
        
        Args:
            x_account_name: X account name (e.g., 'podcasts_tldr')
            
        Returns:
            Telegram channel name or None
        """
        # Simple mapping: add '_tg' suffix
        tg_channel = f"{x_account_name}_tg"
        if tg_channel in self.publishers:
            return tg_channel
        
        # Fallback: try to find any enabled channel
        for channel_name in self.publishers:
            return channel_name
        
        return None
    
    def post_thread(self, thread_record: Dict) -> bool:
        """
        Post a thread to Telegram.
        
        Args:
            thread_record: Thread data from ThreadQueue
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine which Telegram channel to use
            x_account = thread_record.get('account_name', '')
            channel_name = self.get_channel_for_account(x_account)
            
            if not channel_name:
                logger.warning(f"No Telegram channel configured for account: {x_account}")
                return False
            
            publisher = self.publishers.get(channel_name)
            if not publisher:
                logger.error(f"Publisher not found for channel: {channel_name}")
                return False
            
            # Get thread content
            tweets = thread_record.get('tweets', [])
            if isinstance(tweets, str):
                tweets = json.loads(tweets)
            
            podcast_name = thread_record.get('podcast_name', 'Podcast')
            episode_title = thread_record.get('episode_title', 'Episode')
            thumbnail_path = thread_record.get('thumbnail_path')
            
            logger.info(f"Posting to Telegram ({channel_name}): {podcast_name} - {episode_title}")
            
            # Publish
            result = publisher.publish_thread(
                tweets=tweets,
                podcast_name=podcast_name,
                episode_title=episode_title,
                thumbnail_path=thumbnail_path,
                account_name=channel_name
            )
            
            if result.get('success'):
                # Update thread status
                self.thread_queue.update_thread_status(
                    thread_record['thread_id'],
                    'posted',
                    posted_time=datetime.now().isoformat()
                )
                logger.info(f"Successfully posted to Telegram: {result.get('message_id')}")
                return True
            else:
                logger.error(f"Failed to post to Telegram: {result.get('error')}")
                self.thread_queue.update_thread_status(
                    thread_record['thread_id'],
                    'failed'
                )
                return False
                
        except Exception as e:
            logger.error(f"Error posting to Telegram: {e}")
            return False
    
    def process_ready_threads(self) -> int:
        """
        Process all ready Telegram threads.
        
        Returns:
            Number of threads posted
        """
        if not self.publishers:
            return 0
        
        ready_threads = self.thread_queue.get_ready_threads(platform='telegram')
        posted_count = 0
        
        for thread in ready_threads:
            if self.post_thread(thread):
                posted_count += 1
        
        if posted_count > 0:
            logger.info(f"Posted {posted_count} threads to Telegram")
        
        return posted_count
    
    def test_all_channels(self) -> Dict[str, bool]:
        """
        Test connection to all configured channels.

        Returns:
            Dict of channel_name -> connection_success
        """
        results = {}

        for channel_name, publisher in self.publishers.items():
            results[channel_name] = publisher.test_connection()

        return results

    def post_thread_immediately(self,
                                thread_tweets: List[str],
                                podcast_name: str,
                                episode_title: str,
                                thumbnail_path: str = None,
                                x_account_name: str = "podcasts_tldr") -> bool:
        """
        Post to Telegram immediately without queue.

        This bypasses the thread queue and posts directly to Telegram,
        allowing Telegram posts to happen independently of X rate limits.

        Args:
            thread_tweets: List of tweet strings (will be combined into one message)
            podcast_name: Name of the podcast
            episode_title: Title of the episode
            thumbnail_path: Optional path to thumbnail image
            x_account_name: X account name to map to Telegram channel

        Returns:
            True if successful, False otherwise
        """
        if not self.publishers:
            logger.warning("No Telegram publishers configured")
            return False

        channel_name = self.get_channel_for_account(x_account_name)
        if not channel_name:
            logger.warning(f"No Telegram channel for account: {x_account_name}")
            return False

        publisher = self.publishers.get(channel_name)
        if not publisher:
            logger.error(f"Publisher not found: {channel_name}")
            return False

        try:
            logger.info(f"Posting immediately to Telegram ({channel_name}): {podcast_name}")

            result = publisher.publish_thread(
                tweets=thread_tweets,
                podcast_name=podcast_name,
                episode_title=episode_title,
                thumbnail_path=thumbnail_path,
                account_name=channel_name
            )

            if result.get('success'):
                logger.info(f"Immediate Telegram post success: {result.get('message_id')}")
                return True
            else:
                logger.error(f"Immediate Telegram post failed: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"Error in immediate Telegram post: {e}")
            return False


def create_telegram_scheduler_from_config(config_file: str = "viral_config.json",
                                           thread_queue: ThreadQueue = None) -> Optional[TelegramScheduler]:
    """
    Create TelegramScheduler from config file.
    
    Args:
        config_file: Path to viral_config.json
        thread_queue: Optional ThreadQueue instance
        
    Returns:
        TelegramScheduler or None if not configured
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        telegram_config = config.get('telegram_channels', {})
        
        if not telegram_config:
            logger.info("No Telegram channels configured")
            return None
        
        # Check if any channel is enabled
        enabled_channels = [k for k, v in telegram_config.items() if v.get('enabled')]
        if not enabled_channels:
            logger.info("No Telegram channels enabled")
            return None
        
        return TelegramScheduler(telegram_config, thread_queue)
        
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_file}")
        return None
    except Exception as e:
        logger.error(f"Error creating TelegramScheduler: {e}")
        return None
