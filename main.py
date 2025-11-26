#!/usr/bin/env python3
"""Main orchestrator for the podcast to X.com tweets pipeline."""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

import os
from src.config import Config
from src.rss_parser import RSSFeedParser, PodcastEpisode
from src.multi_transcriber import MultiTranscriber
from src.ai_analyzer import AIAnalyzer
from src.thread_generator import ThreadGenerator, ThreadStyle
from src.x_publisher import MultiAccountPublisher
from src.podcast_summary_generator import PodcastSummaryGenerator


# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

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
    
    def __init__(self, config: Config, local_whisper_model: str = "base"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration object
            local_whisper_model: Local Whisper model size
        """
        self.config = config
        
        self.rss_parser = RSSFeedParser(
            feed_urls=config.get_podcast_feeds(),
            days_back=config.days_back
        )
        
        self.transcriber = MultiTranscriber(
            openai_api_key=config.openai_api_key,
            whisper_model=config.whisper_model,
            local_whisper_model=local_whisper_model,
            preferred_methods=getattr(config, 'transcription_methods', None)
        )
        
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
        
        # Check for OpenRouter configuration
        use_openrouter = os.getenv("USE_OPENROUTER", "").lower() == "true"
        if use_openrouter:
            api_key = os.getenv("OPENROUTER_API_KEY", config.openai_api_key)
            base_url = "https://openrouter.ai/api/v1"
            model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
            logger.info("Using OpenRouter API for summary generation")
        else:
            api_key = config.openai_api_key
            base_url = None
            model = config.gpt_model
        
        # Summary generator for 1000-word summaries (stored locally, NOT uploaded)
        self.summary_generator = PodcastSummaryGenerator(
            openai_api_key=api_key,
            model=model,
            base_url=base_url,
            db_path="data/summaries.db"
        )
    
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
        if not transcription or transcription.get('error'):
            logger.error(f"❌ Transcription failed for episode: {episode.title}")
            if transcription and transcription.get('attempts'):
                for attempt in transcription['attempts']:
                    logger.error(f"  - {attempt['method']}: {attempt.get('status', 'unknown')} - {attempt.get('error', 'No error details')}")
            logger.info(f"⏭️  Skipping episode '{episode.title}' due to transcription failure")
            return
        
        # Generate 1000-word summary (stored locally, NOT uploaded to X.com)
        self._generate_summary(episode, transcription)
        
        highlights = self._analyze_episode(episode, transcription)
        if not highlights:
            logger.warning(f"No highlights extracted for: {episode.title}")
            return
        
        threads = self._generate_threads(episode, highlights)
        
        results = self._publish_threads(threads, episode)
        
        self._log_results(episode, results)
    
    def _transcribe_episode(self, episode: PodcastEpisode) -> Dict[str, Any]:
        """Transcribe episode using best available method."""
        logger.info(f"Transcribing episode: {episode.title}")
        
        return self.transcriber.transcribe_episode(
            audio_url=episode.audio_url,
            youtube_urls=episode.youtube_urls or [],
            title=episode.title
        )
    
    def _generate_summary(self, episode: PodcastEpisode, transcription: Dict[str, Any]):
        """Generate 1000-word summary (stored locally, NOT uploaded to X.com)."""
        logger.info(f"Generating comprehensive summary for: {episode.title}")
        
        try:
            # Create a unique episode ID from the episode data
            import hashlib
            episode_id = hashlib.md5(
                f"{episode.podcast_name}:{episode.title}:{episode.published_date.isoformat()}".encode()
            ).hexdigest()[:12]
            
            summary = self.summary_generator.generate_summary(
                transcription_text=transcription.get("text", ""),
                podcast_name=episode.podcast_name,
                episode_title=episode.title,
                episode_id=episode_id,
                episode_number=episode.episode_number,
                published_date=episode.published_date.isoformat() if episode.published_date else None,
                transcription_method=transcription.get("method", "unknown")
            )
            
            if summary:
                logger.info(f"✅ Generated {summary.word_count}-word summary (stored locally)")
            else:
                logger.warning(f"⚠️ Summary generation failed for: {episode.title}")
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Don't fail the whole processing, just log and continue
    
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
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Local Whisper model size (default: base)"
    )
    
    # Summary-related arguments
    parser.add_argument(
        "--list-summaries",
        action="store_true",
        help="List all stored podcast summaries"
    )
    parser.add_argument(
        "--summary-stats",
        action="store_true",
        help="Show summary database statistics"
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
    
    pipeline = PodcastTweetsPipeline(config, args.whisper_model)
    
    # Handle summary-specific commands
    if args.list_summaries:
        summaries = pipeline.summary_generator.list_summaries(limit=50)
        if summaries:
            print("\n📝 STORED PODCAST SUMMARIES")
            print("=" * 80)
            for i, s in enumerate(summaries, 1):
                print(f"{i}. [{s.podcast_name}] {s.episode_title[:50]}... ({s.word_count} words)")
            print(f"\nTotal: {len(summaries)} summaries in database")
        else:
            print("No summaries found in database.")
        sys.exit(0)
    
    if args.summary_stats:
        stats = pipeline.summary_generator.get_stats()
        print("\n📊 SUMMARY DATABASE STATISTICS")
        print("=" * 50)
        print(f"Total Summaries: {stats.get('total_summaries', 0)}")
        print(f"Average Word Count: {int(stats.get('average_word_count', 0))}")
        print("\nSummaries by Podcast:")
        for podcast, count in stats.get('by_podcast', {}).items():
            print(f"  • {podcast}: {count}")
        print("\nRecent Summaries:")
        for recent in stats.get('recent_summaries', []):
            print(f"  • {recent['title'][:40]}... ({recent['podcast']})")
        sys.exit(0)
    
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