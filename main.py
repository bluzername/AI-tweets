#!/usr/bin/env python3
"""Main orchestrator for the podcast to X.com tweets pipeline."""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

from src.config import Config
from src.rss_parser import RSSFeedParser, PodcastEpisode
from src.transcriber import WhisperTranscriber, TranscriptionCache
from src.ai_analyzer import AIAnalyzer
from src.thread_generator import ThreadGenerator, ThreadStyle
from src.x_publisher import MultiAccountPublisher


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PodcastTweetsPipeline:
    """Main orchestrator for the podcast to tweets pipeline."""
    
    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        self.rss_parser = RSSFeedParser(
            feed_urls=config.get_podcast_feeds(),
            days_back=config.days_back
        )
        
        self.transcriber = WhisperTranscriber(api_key=config.openai_api_key)
        self.transcription_cache = TranscriptionCache()
        
        self.analyzer = AIAnalyzer(
            api_key=config.openai_api_key,
            model=config.gpt_model
        )
        
        self.thread_generator = ThreadGenerator(
            api_key=config.openai_api_key,
            model=config.gpt_model
        )
        
        accounts_config = {
            name: acc.get_credentials()
            for name, acc in config.accounts.items()
        }
        self.publisher = MultiAccountPublisher(accounts_config)
    
    def run(self, episode_limit: int = None, specific_podcast: str = None):
        """
        Run the complete pipeline.
        
        Args:
            episode_limit: Maximum number of episodes to process
            specific_podcast: Process only this podcast
        """
        logger.info("Starting podcast to tweets pipeline")
        
        episodes = self._fetch_episodes(specific_podcast)
        
        if episode_limit:
            episodes = episodes[:episode_limit]
        
        logger.info(f"Processing {len(episodes)} episodes")
        
        for episode in episodes:
            try:
                self._process_episode(episode)
            except Exception as e:
                logger.error(f"Error processing episode {episode.title}: {e}")
                continue
        
        logger.info("Pipeline completed")
    
    def _fetch_episodes(self, specific_podcast: str = None) -> List[PodcastEpisode]:
        """Fetch recent podcast episodes."""
        logger.info("Fetching recent podcast episodes")
        
        if specific_podcast:
            podcast_config = self.config.get_podcast_by_name(specific_podcast)
            if podcast_config:
                parser = RSSFeedParser(
                    feed_urls=[podcast_config.rss_url],
                    days_back=self.config.days_back
                )
                return parser.parse_feeds()
            else:
                logger.error(f"Podcast not found: {specific_podcast}")
                return []
        
        return self.rss_parser.parse_feeds()
    
    def _process_episode(self, episode: PodcastEpisode):
        """Process a single podcast episode."""
        logger.info(f"Processing: {episode}")
        
        transcription = self._transcribe_episode(episode)
        if not transcription:
            logger.error(f"Failed to transcribe episode: {episode.title}")
            return
        
        highlights = self._analyze_episode(episode, transcription)
        if not highlights:
            logger.warning(f"No highlights extracted for: {episode.title}")
            return
        
        threads = self._generate_threads(episode, highlights)
        
        results = self._publish_threads(threads, episode)
        
        self._log_results(episode, results)
    
    def _transcribe_episode(self, episode: PodcastEpisode) -> Dict[str, Any]:
        """Transcribe episode audio."""
        cached = self.transcription_cache.get(episode.audio_url)
        if cached:
            logger.info(f"Using cached transcription for {episode.title}")
            return cached
        
        logger.info(f"Transcribing audio for {episode.title}")
        transcription = self.transcriber.transcribe_audio(episode.audio_url)
        
        if transcription:
            self.transcription_cache.set(episode.audio_url, transcription)
        
        return transcription
    
    def _analyze_episode(self, episode: PodcastEpisode, transcription: Dict[str, Any]) -> List:
        """Analyze episode transcription for highlights."""
        logger.info(f"Analyzing transcription for {episode.title}")
        
        return self.analyzer.extract_highlights(
            transcription=transcription.get("text", ""),
            podcast_name=episode.podcast_name,
            episode_title=episode.title,
            max_highlights=self.config.max_highlights
        )
    
    def _generate_threads(self, episode: PodcastEpisode, highlights: List) -> Dict[str, List]:
        """Generate threads for each configured account."""
        threads = {}
        
        podcast_config = self.config.get_podcast_by_name(episode.podcast_name)
        podcast_handles = {
            "podcast": podcast_config.x_handle if podcast_config else None,
            "creators": podcast_config.creator_handles if podcast_config else []
        }
        
        for account_name, account_config in self.config.accounts.items():
            logger.info(f"Generating thread for account: {account_name}")
            
            style = ThreadStyle(
                tone=account_config.tone,
                emoji_usage=account_config.emoji_usage,
                hashtag_style=account_config.hashtag_style,
                cta_style="medium"
            )
            
            scored_highlights = self.analyzer.score_highlights(
                highlights, account_config.target_audience
            )
            
            thread = self.thread_generator.generate_thread(
                highlights=scored_highlights,
                podcast_name=episode.podcast_name,
                episode_title=episode.title,
                episode_number=episode.episode_number,
                style=style,
                target_audience=account_config.target_audience,
                podcast_handles=podcast_handles
            )
            
            adapted_thread = self.thread_generator.adapt_thread_to_account(
                thread,
                account_config.to_dict()
            )
            
            threads[account_name] = adapted_thread
        
        return threads
    
    def _publish_threads(self, threads: Dict, episode: PodcastEpisode) -> Dict:
        """Publish threads to X.com or fallback to markdown."""
        logger.info(f"Publishing threads for {episode.title}")
        
        return self.publisher.publish_to_all(
            threads=threads,
            podcast_name=episode.podcast_name,
            episode_title=episode.title,
            dry_run=self.config.dry_run
        )
    
    def _log_results(self, episode: PodcastEpisode, results: Dict):
        """Log publishing results."""
        logger.info(f"Results for {episode.title}:")
        for account, result in results.items():
            status = result.get("status", "unknown")
            if status == "success":
                logger.info(f"  {account}: Successfully posted to X.com")
            elif status == "fallback":
                logger.info(f"  {account}: Saved to {result.get('file_path')}")
            else:
                logger.error(f"  {account}: Failed - {result.get('message')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert podcast episodes to X.com threads"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of episodes to process"
    )
    parser.add_argument(
        "--podcast",
        help="Process only this specific podcast"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually post to X.com"
    )
    
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    config = Config(config_file=args.config, env_file=args.env)
    
    if args.dry_run:
        config.dry_run = True
    
    if not config.validate():
        logger.error("Invalid configuration. Please check your settings.")
        sys.exit(1)
    
    pipeline = PodcastTweetsPipeline(config)
    
    try:
        pipeline.run(
            episode_limit=args.limit,
            specific_podcast=args.podcast
        )
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()