#!/usr/bin/env python3
"""
Telegram Publisher - Posts content to Telegram channels via Bot API.
Much simpler than X API - no rate limits, no OAuth complexity.
"""

import logging
import re
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramPublisher:
    """
    Handles publishing threads to Telegram channels.
    
    Features:
    - Simple Bot API (just HTTP requests)
    - No rate limits for channel posts
    - Supports photos with captions
    - Markdown formatting
    """
    
    def __init__(self, bot_token: str, channel_id: str):
        """
        Initialize Telegram publisher.
        
        Args:
            bot_token: Bot token from @BotFather
            channel_id: Channel username (@channel) or numeric ID
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Validate token format
        if not bot_token or ':' not in bot_token:
            logger.warning("Invalid bot token format. Expected: 123456789:ABCdefGHI...")
        
        logger.info(f"TelegramPublisher initialized for channel: {channel_id}")
    
    def _make_request(self, method: str, data: Dict = None, files: Dict = None) -> Dict:
        """Make a request to Telegram Bot API."""
        url = f"{self.api_url}/{method}"
        
        try:
            if files:
                response = requests.post(url, data=data, files=files, timeout=30)
            else:
                response = requests.post(url, json=data, timeout=30)
            
            result = response.json()
            
            if not result.get('ok'):
                error_desc = result.get('description', 'Unknown error')
                logger.error(f"Telegram API error: {error_desc}")
                return {'ok': False, 'error': error_desc}
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Request to Telegram failed: {e}")
            return {'ok': False, 'error': str(e)}
    
    def send_message(self, text: str, parse_mode: str = 'Markdown',
                     disable_preview: bool = True) -> Dict:
        """
        Send a text message to the channel.
        
        Args:
            text: Message text (up to 4096 characters)
            parse_mode: 'Markdown' or 'HTML'
            disable_preview: Disable link previews
            
        Returns:
            API response dict
        """
        data = {
            'chat_id': self.channel_id,
            'text': text,
            'parse_mode': parse_mode,
            'disable_web_page_preview': disable_preview
        }
        
        return self._make_request('sendMessage', data)
    
    def send_photo(self, photo_path: str, caption: str = None,
                   parse_mode: str = 'Markdown', reply_to_message_id: int = None) -> Dict:
        """
        Send a photo with optional caption.

        Args:
            photo_path: Path to image file
            caption: Photo caption (up to 1024 characters)
            parse_mode: 'Markdown' or 'HTML'
            reply_to_message_id: Optional message ID to reply to

        Returns:
            API response dict
        """
        data = {
            'chat_id': self.channel_id,
            'parse_mode': parse_mode
        }

        if caption:
            # Telegram caption limit is 1024 chars
            if len(caption) > 1024:
                caption = caption[:1021] + '...'
            data['caption'] = caption

        if reply_to_message_id:
            data['reply_to_message_id'] = reply_to_message_id

        try:
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                return self._make_request('sendPhoto', data, files)
        except FileNotFoundError:
            logger.error(f"Photo not found: {photo_path}")
            return {'ok': False, 'error': f'Photo not found: {photo_path}'}
    
    def _clean_tweet_for_telegram(self, tweet: str) -> str:
        """Remove X-specific formatting from tweet text."""
        text = tweet
        # Remove emoji numbers (1️⃣, 2️⃣, etc.)
        text = re.sub(r'[0-9]️⃣', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove thread markers (🧵, /thread, etc.)
        text = re.sub(r'🧵|/thread|Thread:', '', text, flags=re.IGNORECASE)
        # Remove leading bullet markers like "1.", "2." etc that may be in the tweet
        text = re.sub(r'^\s*\d+\.\s*', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _get_dynamic_hashtags(self, podcast_name: str) -> str:
        """Generate relevant hashtags based on podcast category."""
        podcast_tags = {
            "Lex Fridman": "#AI #science #podcast",
            "Huberman Lab": "#health #science #neuroscience",
            "All-In": "#business #tech #startups",
            "Acquired": "#business #startups #investing",
            "Tim Ferriss": "#productivity #business #podcast",
            "Diary Of A CEO": "#business #entrepreneurship #podcast",
            "Syntax": "#webdev #coding #javascript",
            "Vergecast": "#tech #gadgets #news",
            "Pivot": "#tech #business #media",
            "Pragmatic Engineer": "#engineering #tech #career",
            "20VC": "#vc #startups #investing",
            "Twenty Minute VC": "#vc #startups #investing",
            "Joe Rogan": "#podcast #culture #interviews",
            "Masters of Scale": "#business #entrepreneurship #startups",
        }
        for key, tags in podcast_tags.items():
            if key.lower() in podcast_name.lower():
                return tags
        return "#podcast #tldr"

    def format_thread_as_message(self, tweets: List[str], podcast_name: str,
                                  episode_title: str,
                                  deep_links: Dict[str, str] = None,
                                  episode_url: str = None) -> str:
        """
        Convert X thread format to single Telegram message.

        Args:
            tweets: List of tweet texts
            podcast_name: Name of the podcast
            episode_title: Episode title
            deep_links: Optional dict with 'spotify' and/or 'apple' URLs
            episode_url: Optional direct episode URL

        Returns:
            Formatted message string
        """
        # Header
        header = f"🎙️ *{podcast_name}*\n📺 _{episode_title}_\n"

        # Format as simple numbered list (NO separators, NO emoji numbers)
        points_text = ""
        for i, tweet in enumerate(tweets, 1):
            clean_text = self._clean_tweet_for_telegram(tweet)
            if clean_text:  # Only add non-empty points
                points_text += f"\n{i}. {clean_text}\n"

        # Add links section at bottom
        links_section = ""
        if deep_links:
            parts = []
            if 'spotify' in deep_links:
                parts.append(f"[Spotify]({deep_links['spotify']})")
            if 'apple' in deep_links:
                parts.append(f"[Apple]({deep_links['apple']})")
            if parts:
                links_section = f"\n🎧 Listen: {' | '.join(parts)}"
        elif episode_url:
            links_section = f"\n🎧 [Listen to full episode]({episode_url})"

        # Dynamic hashtags based on podcast category
        hashtags = self._get_dynamic_hashtags(podcast_name)
        hashtag_section = f"\n\n{hashtags}"

        # Combine
        message = f"{header}{points_text}{links_section}{hashtag_section}"

        # Telegram message limit is 4096 characters
        if len(message) > 4096:
            # Truncate points, keep header and footer
            footer = f"{links_section}{hashtag_section}"
            max_points = 4096 - len(header) - len(footer) - 50
            points_text = points_text[:max_points]
            # Cut at last complete point
            last_newline = points_text.rfind('\n\n')
            if last_newline > 0:
                points_text = points_text[:last_newline]
            message = f"{header}{points_text}\n\n_(more insights in full episode...)_{footer}"

        return message
    
    def publish_thread(self, tweets: List[str], podcast_name: str,
                       episode_title: str, thumbnail_path: str = None,
                       account_name: str = None,
                       deep_links: Dict[str, str] = None,
                       episode_url: str = None) -> Dict[str, Any]:
        """
        Publish a thread to Telegram channel.

        Args:
            tweets: List of tweet texts forming the thread
            podcast_name: Name of the podcast
            episode_title: Episode title
            thumbnail_path: Optional path to thumbnail image
            account_name: Account name (for logging)
            deep_links: Optional dict with 'spotify' and/or 'apple' URLs
            episode_url: Optional direct episode URL

        Returns:
            Dictionary with publish status and details
        """
        logger.info(f"Publishing to Telegram: {podcast_name} - {episode_title}")

        # Format the message with deep links and hashtags
        message = self.format_thread_as_message(
            tweets, podcast_name, episode_title,
            deep_links=deep_links, episode_url=episode_url
        )
        
        result = {
            'platform': 'telegram',
            'channel_id': self.channel_id,
            'podcast_name': podcast_name,
            'episode_title': episode_title,
            'tweet_count': len(tweets),
            'timestamp': datetime.now().isoformat()
        }
        
        has_thumbnail = thumbnail_path and Path(thumbnail_path).exists()

        # If message fits in photo caption (1024 chars), send as single photo+caption
        if has_thumbnail and len(message) <= 1024:
            response = self.send_photo(thumbnail_path, message)
            if response.get('ok'):
                result['success'] = True
                result['message_id'] = response['result']['message_id']
                result['has_photo'] = True
                logger.info(f"Posted to Telegram with photo: {result['message_id']}")
                return result

        # Otherwise, send text first, then photo as reply (keeps them connected)
        response = self.send_message(message)

        if response.get('ok'):
            result['success'] = True
            result['message_id'] = response['result']['message_id']
            logger.info(f"Posted to Telegram: {result['message_id']}")

            # Send thumbnail as reply to the text message
            if has_thumbnail:
                photo_response = self.send_photo(
                    thumbnail_path,
                    reply_to_message_id=result['message_id']
                )
                if photo_response.get('ok'):
                    result['has_photo'] = True
                    result['photo_message_id'] = photo_response['result']['message_id']
                    logger.info(f"Added photo as reply: {result['photo_message_id']}")
        else:
            result['success'] = False
            result['error'] = response.get('error', 'Unknown error')
            logger.error(f"Failed to post to Telegram: {result['error']}")
        
        return result
    
    def test_connection(self) -> bool:
        """Test if the bot can access the channel."""
        response = self._make_request('getChat', {'chat_id': self.channel_id})
        
        if response.get('ok'):
            chat = response['result']
            logger.info(f"Connected to channel: {chat.get('title', self.channel_id)}")
            return True
        else:
            logger.error(f"Cannot access channel: {response.get('error')}")
            return False


# Test function
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 3:
        print("Usage: python telegram_publisher.py <bot_token> <channel_id>")
        print("Example: python telegram_publisher.py 123:ABC @mychannel")
        sys.exit(1)
    
    bot_token = sys.argv[1]
    channel_id = sys.argv[2]
    
    publisher = TelegramPublisher(bot_token, channel_id)
    
    if publisher.test_connection():
        print("✅ Connection successful!")
        
        # Test post
        test_tweets = [
            "🧵 This is a test thread from Podcast TLDR!",
            "Here's an interesting insight from a recent podcast episode.",
            "And here's the conclusion. Follow for more!"
        ]
        
        result = publisher.publish_thread(
            tweets=test_tweets,
            podcast_name="Test Podcast",
            episode_title="Test Episode"
        )
        
        if result.get('success'):
            print(f"✅ Test post successful! Message ID: {result['message_id']}")
        else:
            print(f"❌ Test post failed: {result.get('error')}")
    else:
        print("❌ Connection failed!")
