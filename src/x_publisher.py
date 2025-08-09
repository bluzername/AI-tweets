"""X.com (Twitter) publishing module with markdown fallback."""

import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import tweepy
from .thread_generator import Tweet

logger = logging.getLogger(__name__)


class XPublisher:
    """Handles publishing threads to X.com with fallback to markdown."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
        fallback_dir: str = "output/threads"
    ):
        """
        Initialize X publisher.
        
        Args:
            api_key: X API key
            api_secret: X API secret
            access_token: X access token
            access_token_secret: X access token secret
            bearer_token: X bearer token
            fallback_dir: Directory for markdown fallback files
        """
        self.client = None
        self.fallback_dir = Path(fallback_dir)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        
        if all([api_key, api_secret, access_token, access_token_secret]):
            try:
                self.client = tweepy.Client(
                    bearer_token=bearer_token,
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    wait_on_rate_limit=True
                )
                logger.info("X.com client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize X client: {e}")
                self.client = None
        else:
            logger.warning("X.com credentials not provided. Will use markdown fallback.")
    
    def publish_thread(
        self,
        thread: List[Tweet],
        podcast_name: str,
        episode_title: str,
        account_name: str = "default",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Publish thread to X.com or fallback to markdown.
        
        Args:
            thread: List of tweets forming the thread
            podcast_name: Name of the podcast
            episode_title: Episode title
            account_name: Name of the account posting
            dry_run: If True, don't actually post
            
        Returns:
            Dictionary with publish status and details
        """
        if dry_run:
            logger.info("Dry run mode - not posting to X.com")
            return self._save_to_markdown(thread, podcast_name, episode_title, account_name)
        
        if self.client:
            try:
                return self._post_to_x(thread, account_name)
            except Exception as e:
                logger.error(f"Failed to post to X.com: {e}")
                logger.info("Falling back to markdown output")
                return self._save_to_markdown(thread, podcast_name, episode_title, account_name)
        else:
            logger.info("No X.com client available. Saving to markdown.")
            return self._save_to_markdown(thread, podcast_name, episode_title, account_name)
    
    def _post_to_x(self, thread: List[Tweet], account_name: str) -> Dict[str, Any]:
        """
        Post thread to X.com.
        
        Args:
            thread: List of tweets
            account_name: Account name for logging
            
        Returns:
            Status dictionary
        """
        posted_tweets = []
        previous_tweet_id = None
        
        try:
            for tweet in thread:
                if previous_tweet_id:
                    response = self.client.create_tweet(
                        text=tweet.content,
                        in_reply_to_tweet_id=previous_tweet_id
                    )
                else:
                    response = self.client.create_tweet(text=tweet.content)
                
                tweet_id = response.data['id']
                posted_tweets.append({
                    "position": tweet.position,
                    "tweet_id": tweet_id,
                    "content": tweet.content
                })
                previous_tweet_id = tweet_id
                
                logger.info(f"Posted tweet {tweet.position} for {account_name}")
            
            return {
                "status": "success",
                "platform": "x.com",
                "account": account_name,
                "tweets_posted": len(posted_tweets),
                "thread_url": f"https://x.com/{account_name}/status/{posted_tweets[0]['tweet_id']}",
                "tweet_ids": [t['tweet_id'] for t in posted_tweets]
            }
            
        except Exception as e:
            logger.error(f"Error posting thread: {e}")
            
            if posted_tweets:
                logger.warning(f"Partial thread posted: {len(posted_tweets)} tweets")
            
            raise
    
    def _save_to_markdown(
        self,
        thread: List[Tweet],
        podcast_name: str,
        episode_title: str,
        account_name: str
    ) -> Dict[str, Any]:
        """
        Save thread to markdown file for manual posting.
        
        Args:
            thread: List of tweets
            podcast_name: Podcast name
            episode_title: Episode title
            account_name: Account name
            
        Returns:
            Status dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{account_name}_{podcast_name.replace(' ', '_')}.md"
        filepath = self.fallback_dir / filename
        
        content = self._format_thread_markdown(thread, podcast_name, episode_title, account_name)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Thread saved to {filepath}")
        
        return {
            "status": "fallback",
            "platform": "markdown",
            "account": account_name,
            "file_path": str(filepath),
            "tweets_count": len(thread),
            "message": "Thread saved to markdown file for manual posting"
        }
    
    def _format_thread_markdown(
        self,
        thread: List[Tweet],
        podcast_name: str,
        episode_title: str,
        account_name: str
    ) -> str:
        """
        Format thread as markdown for easy copy-paste.
        
        Args:
            thread: List of tweets
            podcast_name: Podcast name
            episode_title: Episode title
            account_name: Account name
            
        Returns:
            Formatted markdown string
        """
        lines = [
            f"# X.com Thread - {podcast_name}",
            f"**Episode:** {episode_title}",
            f"**Account:** {account_name}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Thread Content",
            "",
            "*Copy and paste each section below as a separate tweet in your thread:*",
            ""
        ]
        
        for i, tweet in enumerate(thread, 1):
            lines.extend([
                f"### Tweet {i}/{len(thread)}",
                "",
                "```",
                tweet.content,
                "```",
                "",
                f"*Characters: {len(tweet.content)}/280*",
                ""
            ])
            
            if i < len(thread):
                lines.extend(["---", "[Thread Break]", "---", ""])
        
        lines.extend([
            "",
            "---",
            "",
            "## Posting Instructions",
            "",
            "1. Copy the first tweet and post it on X.com",
            "2. Reply to your own tweet with the second tweet",
            "3. Continue replying to create the thread",
            "4. Make sure to maintain the thread by always replying to the previous tweet",
            "",
            "## Character Counts",
            ""
        ])
        
        for i, tweet in enumerate(thread, 1):
            lines.append(f"- Tweet {i}: {len(tweet.content)}/280 characters")
        
        return "\n".join(lines)


class MultiAccountPublisher:
    """Manages publishing to multiple X.com accounts."""
    
    def __init__(self, accounts_config: Dict[str, Dict[str, Any]]):
        """
        Initialize multi-account publisher.
        
        Args:
            accounts_config: Dictionary of account configurations
        """
        self.publishers = {}
        
        for account_name, config in accounts_config.items():
            self.publishers[account_name] = XPublisher(
                api_key=config.get('api_key'),
                api_secret=config.get('api_secret'),
                access_token=config.get('access_token'),
                access_token_secret=config.get('access_token_secret'),
                bearer_token=config.get('bearer_token'),
                fallback_dir=config.get('fallback_dir', f"output/{account_name}")
            )
            logger.info(f"Initialized publisher for account: {account_name}")
    
    def publish_to_account(
        self,
        account_name: str,
        thread: List[Tweet],
        podcast_name: str,
        episode_title: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Publish thread to specific account.
        
        Args:
            account_name: Name of the account
            thread: Thread to publish
            podcast_name: Podcast name
            episode_title: Episode title
            dry_run: If True, don't actually post
            
        Returns:
            Publish status
        """
        if account_name not in self.publishers:
            logger.error(f"Account {account_name} not configured")
            return {
                "status": "error",
                "message": f"Account {account_name} not found"
            }
        
        return self.publishers[account_name].publish_thread(
            thread, podcast_name, episode_title, account_name, dry_run
        )
    
    def publish_to_all(
        self,
        threads: Dict[str, List[Tweet]],
        podcast_name: str,
        episode_title: str,
        dry_run: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Publish to all configured accounts.
        
        Args:
            threads: Dictionary of account_name -> thread
            podcast_name: Podcast name
            episode_title: Episode title
            dry_run: If True, don't actually post
            
        Returns:
            Dictionary of publish statuses per account
        """
        results = {}
        
        for account_name, thread in threads.items():
            if account_name in self.publishers:
                results[account_name] = self.publish_to_account(
                    account_name, thread, podcast_name, episode_title, dry_run
                )
            else:
                logger.warning(f"No thread provided for account: {account_name}")
        
        return results