#!/usr/bin/env python3
"""
Hebrew Orchestrator - Autonomous Pipeline for Hebrew Podcast TLDR
Processes Hebrew podcasts and publishes to Telegram only.
"""

import logging
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import os

from src.config import PipelineConfig
from src.telegram_publisher import TelegramPublisher
from src.podcast_ingestor import PodcastIngestor, EpisodeDatabase
from src.viral_transcriber import ViralTranscriber
from src.viral_insight_extractor import ViralContentAnalyzer

logger = logging.getLogger(__name__)


class HebrewOrchestrator:
    """
    Autonomous orchestrator for Hebrew podcast pipeline.

    Features:
    - Hebrew podcast RSS monitoring
    - Whisper transcription with Hebrew language
    - Hebrew prompts for content generation
    - Telegram-only publishing
    - Separate database from English pipeline
    """

    def __init__(self, config_file: str = "hebrew_config.json"):
        """Initialize Hebrew orchestrator.

        Args:
            config_file: Path to Hebrew config JSON
        """
        self.config_file = config_file
        self.config = PipelineConfig(config_file)
        self.running = False

        # Setup logging
        self._setup_logging()

        # Initialize components
        self._init_components()

        logger.info(f"Hebrew Orchestrator initialized with config: {config_file}")

    def _setup_logging(self):
        """Setup logging for Hebrew pipeline."""
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/hebrew_orchestrator.log', encoding='utf-8')
            ]
        )

    def _init_components(self):
        """Initialize all pipeline components."""
        # Get feed URLs from config
        feed_urls = []
        for feed_config in self.config.config.get('podcast_feeds', []):
            if isinstance(feed_config, str):
                feed_urls.append(feed_config)
            elif isinstance(feed_config, dict) and 'url' in feed_config:
                feed_urls.append(feed_config['url'])

        # Episode database with Hebrew-specific path
        self.db = EpisodeDatabase(db_path=self.config.episodes_db_path)

        # Podcast ingestor - pass feed URLs (even if empty)
        self.ingestor = PodcastIngestor(
            feed_urls=feed_urls,
            max_episodes_per_feed=20
        )
        # Override the ingestor's database with Hebrew database
        self.ingestor.db = self.db

        # Transcriber with Hebrew language support
        self.transcriber = ViralTranscriber()

        # Content analyzer with Hebrew prompts
        api_key = self.config.openrouter_api_key if self.config.use_openrouter else self.config.openai_api_key
        base_url = "https://openrouter.ai/api/v1" if self.config.use_openrouter else None

        self.analyzer = ViralContentAnalyzer(
            openai_api_key=api_key,
            model="deepseek/deepseek-chat",
            base_url=base_url,
            prompts_dir="/app/" + self.config.prompts_dir
        )

        # Telegram publisher (only output channel)
        if self.config.telegram_enabled and self.config.telegram_bot_token:
            self.telegram = TelegramPublisher(
                bot_token=self.config.telegram_bot_token,
                channel_id=self.config.telegram_channel_id
            )
            logger.info(f"Telegram publisher initialized for {self.config.telegram_channel_id}")
        else:
            self.telegram = None
            logger.warning("Telegram not configured - no output channel!")

    def discover_new_episodes(self) -> int:
        """Discover new Hebrew podcast episodes from configured feeds.

        Returns:
            Number of new episodes discovered
        """
        if not self.ingestor.feed_urls:
            logger.info("No Hebrew podcast feeds configured")
            return 0

        try:
            new_episodes = self.ingestor.discover_new_episodes()
            if new_episodes:
                logger.info(f"Discovered {len(new_episodes)} new Hebrew episodes")
            return len(new_episodes) if new_episodes else 0
        except Exception as e:
            logger.error(f"Error discovering episodes: {e}")
            return 0

    def get_pending_episodes(self, limit: int = 5) -> List[Dict]:
        """Get pending Hebrew episodes to process.

        Args:
            limit: Maximum episodes to return

        Returns:
            List of pending episode dicts
        """
        episodes = self.db.get_pending_episodes(limit=limit)
        # Convert PodcastEpisode objects to dicts
        result = []
        for ep in episodes:
            if hasattr(ep, '__dict__'):
                result.append({
                    'episode_id': ep.episode_id,
                    'title': ep.title,
                    'audio_url': ep.audio_url,
                    'podcast_name': getattr(ep, 'podcast_name', 'Hebrew Podcast'),
                    'guid': getattr(ep, 'guid', ep.episode_id),
                    'episode_url': getattr(ep, 'episode_url', None)
                })
            else:
                result.append(ep)  # Already a dict
        return result

    def process_episode(self, episode: Dict) -> Optional[Dict]:
        """Process a single Hebrew episode.

        Args:
            episode: Episode data dict

        Returns:
            Processing result or None if failed
        """
        episode_id = episode.get('episode_id', episode.get('guid', episode.get('id')))
        title = episode.get('title', 'Unknown')
        podcast_name = episode.get('podcast_name', 'Hebrew Podcast')
        audio_url = episode.get('audio_url')

        logger.info(f"Processing Hebrew episode: {title}")

        # Mark as processing
        self.db.update_episode_status(episode_id, 'processing')

        try:
            if not audio_url:
                logger.error(f"No audio URL for episode: {title}")
                self.db.update_episode_status(episode_id, 'failed', error_message="No audio URL")
                return None

            # 1. Transcribe with Hebrew language
            transcription = self.transcriber.transcribe_for_viral_content(
                audio_url=audio_url,
                youtube_urls=[],
                title=title,
                language="he"  # Hebrew
            )

            if not transcription or not transcription.text:
                logger.error(f"Transcription failed for: {title}")
                self.db.update_episode_status(episode_id, 'failed', error_message="Transcription failed")
                return None

            logger.info(f"Transcribed {len(transcription.text)} chars in Hebrew")

            # 2. Extract key points using Hebrew prompts
            key_points = self.analyzer.extract_key_points(
                transcription=transcription,
                podcast_name=podcast_name,
                episode_title=title,
                num_points=10
            )

            if not key_points:
                logger.warning(f"No key points extracted for: {title}")
                self.db.update_episode_status(episode_id, 'failed', error_message="No key points extracted")
                return None

            logger.info(f"Extracted {len(key_points)} key points in Hebrew")

            # 3. Format for Telegram
            episode_url = episode.get('episode_url')
            message = self._format_hebrew_telegram_post(
                podcast_name=podcast_name,
                episode_title=title,
                key_points=key_points,
                episode_url=episode_url
            )

            # 4. Publish to Telegram
            if self.telegram:
                result = self.telegram.send_message(message)
                if result.get('ok'):
                    message_id = result.get('result', {}).get('message_id')
                    logger.info(f"Posted to Hebrew Telegram: {message_id}")

                    # Mark as completed
                    self.db.update_episode_status(episode_id, 'completed')

                    return {
                        'success': True,
                        'episode_id': episode_id,
                        'title': title,
                        'message_id': message_id
                    }
                else:
                    error = result.get('error', 'Unknown error')
                    logger.error(f"Telegram publish failed: {error}")
                    self.db.update_episode_status(episode_id, 'failed', error_message=f"Telegram: {error}")
            else:
                logger.warning("No Telegram publisher configured")
                self.db.update_episode_status(episode_id, 'failed', error_message="No Telegram configured")

            return None

        except Exception as e:
            logger.error(f"Error processing episode {title}: {e}")
            self.db.update_episode_status(episode_id, 'failed', error_message=str(e))
            return None

    def _format_hebrew_telegram_post(self, podcast_name: str, episode_title: str,
                                      key_points: list, episode_url: str = None) -> str:
        """Format key points into a Hebrew Telegram post.

        Args:
            podcast_name: Name of the podcast
            episode_title: Episode title
            key_points: List of ViralInsight objects

        Returns:
            Formatted Telegram message
        """
        # Hebrew header (RTL friendly)
        header = f"🎙️ *{podcast_name}*\n📺 _{episode_title}_\n"

        # Separator
        separator = "\n─────────────────\n"

        # Format key points
        points_text = ""
        for i, point in enumerate(key_points, 1):
            text = point.text if hasattr(point, 'text') else str(point)
            points_text += f"\n\u200f{i}. {text}\n"  # RLM for RTL display

        # Add episode link if available
        link_text = ""
        if episode_url:
            link_text = f"\n\n🎧 [לפרק המלא]({episode_url})"

        # Combine
        message = f"{header}{separator}{points_text}{link_text}"

        # Telegram limit is 4096 chars
        if len(message) > 4000:
            message = message[:3990] + "\n\n_(המשך...)_"

        return message

    def run_cycle(self) -> Dict[str, int]:
        """Run one processing cycle.

        Returns:
            Dict with processing statistics
        """
        stats = {
            'discovered': 0,
            'processed': 0,
            'failed': 0,
            'posted': 0
        }

        # Discover new episodes
        stats['discovered'] = self.discover_new_episodes()

        # Get pending episodes
        pending = self.get_pending_episodes(limit=5)
        logger.info(f"Found {len(pending)} pending Hebrew episodes")

        for episode in pending:
            result = self.process_episode(episode)
            if result and result.get('success'):
                stats['processed'] += 1
                stats['posted'] += 1
            else:
                stats['failed'] += 1

        return stats

    def run_continuous(self, interval_minutes: int = 60):
        """Run orchestrator continuously.

        Args:
            interval_minutes: Minutes between processing cycles
        """
        self.running = True

        def signal_handler(sig, frame):
            logger.info("Shutdown signal received...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(f"Starting Hebrew orchestrator (interval: {interval_minutes} min)")

        while self.running:
            try:
                cycle_start = datetime.now()

                stats = self.run_cycle()

                logger.info(
                    f"Cycle complete: {stats['discovered']} discovered, "
                    f"{stats['processed']} processed, {stats['posted']} posted"
                )

                # Wait for next cycle
                elapsed = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, interval_minutes * 60 - elapsed)

                if sleep_time > 0 and self.running:
                    logger.info(f"Sleeping {sleep_time/60:.1f} minutes until next cycle...")
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in processing cycle: {e}")
                time.sleep(60)  # Wait 1 minute on error

        logger.info("Hebrew orchestrator stopped")

    def test_connection(self) -> bool:
        """Test Telegram connection.

        Returns:
            True if connection successful
        """
        if not self.telegram:
            logger.error("Telegram not configured")
            return False

        return self.telegram.test_connection()


def main():
    """Main entry point for Hebrew orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description='Hebrew Podcast TLDR Orchestrator')
    parser.add_argument('--config', default='hebrew_config.json',
                       help='Path to Hebrew config file')
    parser.add_argument('--test', action='store_true',
                       help='Test Telegram connection only')
    parser.add_argument('--once', action='store_true',
                       help='Run one cycle and exit')
    parser.add_argument('--interval', type=int, default=60,
                       help='Minutes between cycles (default: 60)')

    args = parser.parse_args()

    orchestrator = HebrewOrchestrator(config_file=args.config)

    if args.test:
        if orchestrator.test_connection():
            print("✅ Telegram connection successful")
        else:
            print("❌ Telegram connection failed")
    elif args.once:
        stats = orchestrator.run_cycle()
        print(f"Cycle complete: {stats}")
    else:
        orchestrator.run_continuous(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
