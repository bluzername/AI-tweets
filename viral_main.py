#!/usr/bin/env python3
"""
Main orchestrator for the Podcasts TLDR Viral Content Machine.
The complete pipeline from podcast ingestion to viral tweet posting.
"""

import logging
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import all viral content components
from src.podcast_ingestor import PodcastIngestor
from src.viral_transcriber import ViralTranscriber
from src.viral_insight_extractor import ViralContentAnalyzer, InsightDatabase
from src.viral_tweet_crafter import ViralTweetCrafter
from src.viral_scheduler import ViralScheduler, PostingStrategy
from src.web_interface import create_app, create_templates

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/viral_pipeline_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PodcastsTLDRPipeline:
    """
    The complete Podcasts TLDR viral content machine.
    
    Architecture:
    1. Podcast Ingestor - Discovers and tracks episodes with viral scoring
    2. Viral Transcriber - Enhanced transcription with speaker ID and timestamps
    3. Insight Extractor - AI-powered viral moment detection
    4. Tweet Crafter - Multi-format viral tweet generation
    5. Scheduler & Publisher - Smart scheduling and posting
    """
    
    def __init__(self, config_file: str = "viral_config.json"):
        """Initialize the complete viral content pipeline."""
        self.config = self._load_config(config_file)
        
        # Initialize data directories
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        Path("output").mkdir(exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        logger.info("🎯 Podcasts TLDR Pipeline initialized successfully!")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration with viral-specific settings."""
        
        default_config = {
            # Podcast sources
            "podcast_feeds": [
                "https://feeds.feedburner.com/venturesstories",
                "https://rss.cnn.com/rss/cnn_topstories.rss",
                "https://feeds.feedburner.com/ycombinator"
            ],
            
            # API keys
            "openai_api_key": "",
            "x_accounts": {
                "podcasts_tldr": {
                    "consumer_key": "",
                    "consumer_secret": "",
                    "access_token": "",
                    "access_token_secret": "",
                    "bearer_token": ""
                }
            },
            
            # Viral optimization settings
            "transcription_methods": ["youtube", "local_whisper", "whisper"],
            "max_insights_per_episode": 8,
            "min_viral_score": 0.4,
            "posting_frequency": {
                "podcasts_tldr": 3  # tweets per day
            },
            
            # Content strategy
            "content_mix": {
                "single_tweets": 0.4,
                "threads": 0.4,
                "polls": 0.1,
                "quote_cards": 0.1
            },
            
            # Scheduling
            "posting_strategy": "optimal_time",
            "timezone": "UTC",
            "optimal_times": {
                "weekday": ["09:00", "12:30", "15:00", "18:00", "20:30"],
                "weekend": ["10:00", "14:00", "19:00"]
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _init_components(self):
        """Initialize all pipeline components."""
        
        # 1. Podcast Ingestor
        self.ingestor = PodcastIngestor(
            feed_urls=self.config["podcast_feeds"],
            max_episodes_per_feed=20
        )
        
        # 2. Viral Transcriber
        self.transcriber = ViralTranscriber(
            openai_api_key=self.config["openai_api_key"],
            preferred_methods=self.config["transcription_methods"],
            enable_speaker_id=True,
            enable_viral_detection=True
        )
        
        # 3. Insight Extractor
        self.insight_analyzer = ViralContentAnalyzer(
            openai_api_key=self.config["openai_api_key"],
            model="gpt-4"
        )
        self.insight_db = InsightDatabase()
        
        # 4. Tweet Crafter
        self.tweet_crafter = ViralTweetCrafter(
            openai_api_key=self.config["openai_api_key"],
            model="gpt-4"
        )
        
        # 5. Scheduler & Publisher
        self.scheduler = ViralScheduler(
            x_accounts=self.config["x_accounts"],
            posting_frequency=self.config["posting_frequency"]
        )
    
    def run_discovery(self) -> List[str]:
        """
        Run podcast discovery and return new episode IDs.
        Step 1: Discover new viral episodes.
        """
        logger.info("🔍 Running podcast discovery...")
        
        new_episodes = self.ingestor.discover_new_episodes()
        
        if not new_episodes:
            logger.info("No new episodes found")
            return []
        
        # Log discoveries
        for episode in new_episodes:
            logger.info(f"📻 Found: {episode.podcast_name} - {episode.title} (viral score: {episode.viral_score:.2f})")
        
        return [ep.episode_id for ep in new_episodes]
    
    def run_processing(self, episode_limit: int = 5) -> Dict[str, Any]:
        """
        Run complete processing pipeline for pending episodes.
        Steps 2-4: Transcribe → Extract insights → Craft tweets.
        """
        logger.info(f"⚡ Running processing pipeline (limit: {episode_limit})...")
        
        # Get episodes for processing
        pending_episodes = self.ingestor.get_episodes_for_processing(episode_limit)
        
        if not pending_episodes:
            logger.info("No episodes pending processing")
            return {"processed": 0, "tweets_created": 0}
        
        total_tweets = 0
        processed_count = 0
        
        for episode in pending_episodes:
            try:
                logger.info(f"🎙️ Processing: {episode.podcast_name} - {episode.title}")
                
                # Mark as processing
                self.ingestor.mark_episode_processing(episode.episode_id)
                
                # Step 2: Transcribe with viral enhancements
                try:
                    transcription = self.transcriber.transcribe_for_viral_content(
                        audio_url=episode.audio_url,
                        youtube_urls=episode.youtube_urls or [],
                        title=episode.title,
                        language=None
                    )
                    
                    if not transcription or not transcription.text:
                        logger.warning(f"❌ Transcription failed for {episode.title}")
                        self.ingestor.mark_episode_failed(episode.episode_id, "Transcription returned empty result")
                        continue
                        
                except Exception as transcription_error:
                    logger.warning(f"❌ Transcription failed for {episode.title}: {transcription_error}")
                    self.ingestor.mark_episode_failed(episode.episode_id, f"Transcription error: {str(transcription_error)}")
                    continue
                
                logger.info(f"✅ Transcribed {len(transcription.segments)} segments, {len(transcription.viral_moments)} viral moments")
                
                # Step 3: Extract viral insights
                insights = self.insight_analyzer.extract_viral_insights(
                    transcription=transcription,
                    podcast_name=episode.podcast_name,
                    episode_title=episode.title,
                    max_insights=self.config["max_insights_per_episode"]
                )
                
                # Filter by viral score threshold
                viral_insights = [
                    insight for insight in insights 
                    if insight.viral_score >= self.config["min_viral_score"]
                ]
                
                if not viral_insights:
                    logger.warning(f"⚠️ No viral insights found for {episode.title}")
                    self.ingestor.mark_episode_completed(episode.episode_id, insights_count=0)
                    continue
                
                logger.info(f"💡 Extracted {len(viral_insights)} viral insights")
                
                # Save insights to database
                self.insight_db.save_insights(viral_insights, episode.episode_id)
                
                # Step 4: Craft viral tweets for each account
                account_tweets = {}
                
                for account_name in self.config["x_accounts"].keys():
                    account_tweets[account_name] = []
                    
                    for insight in viral_insights:
                        # Determine formats based on content mix
                        formats = self._select_formats_for_insight(insight)
                        
                        # Craft tweets
                        tweets = self.tweet_crafter.craft_viral_tweets(
                            insight=insight,
                            podcast_name=episode.podcast_name,
                            episode_title=episode.title,
                            formats=formats
                        )
                        
                        account_tweets[account_name].extend(tweets)
                
                # Step 5: Schedule tweets
                for account_name, tweets in account_tweets.items():
                    if tweets:
                        strategy = PostingStrategy(self.config["posting_strategy"])
                        
                        scheduled_ids = self.scheduler.schedule_tweets(
                            tweets=tweets,
                            account_name=account_name,
                            episode_id=episode.episode_id,
                            podcast_name=episode.podcast_name,
                            strategy=strategy
                        )
                        
                        total_tweets += len(scheduled_ids)
                        logger.info(f"📅 Scheduled {len(scheduled_ids)} tweets for @{account_name}")
                
                # Mark episode as completed
                self.ingestor.mark_episode_completed(
                    episode.episode_id,
                    insights_count=len(viral_insights),
                    tweets_generated=total_tweets
                )
                
                processed_count += 1
                logger.info(f"✅ Completed processing: {episode.title}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {episode.title}: {e}")
                self.ingestor.mark_episode_failed(episode.episode_id, str(e))
                continue
        
        logger.info(f"🎯 Processing complete: {processed_count} episodes, {total_tweets} tweets scheduled")
        
        return {
            "processed": processed_count,
            "tweets_created": total_tweets,
            "episodes": [ep.episode_id for ep in pending_episodes[:processed_count]]
        }
    
    def run_scheduler(self):
        """
        Start the automated scheduler.
        Step 5: Post tweets at optimal times.
        """
        logger.info("🚀 Starting viral tweet scheduler...")
        
        self.scheduler.start_scheduler()
        logger.info("✅ Scheduler running in background")
    
    def run_web_interface(self, port: int = 5000):
        """
        Start the web interface for manual management.
        """
        logger.info(f"🌐 Starting web interface on port {port}...")
        
        # Create templates
        create_templates()
        
        # Create and run Flask app
        app = create_app()
        app.run(debug=False, host="0.0.0.0", port=port)
    
    def run_full_pipeline(self, discovery: bool = True, processing: bool = True, 
                         episode_limit: int = 5) -> Dict[str, Any]:
        """
        Run the complete viral content pipeline.
        """
        logger.info("🎯 Starting complete Podcasts TLDR pipeline...")
        
        results = {
            "discovery": {"new_episodes": 0},
            "processing": {"processed": 0, "tweets_created": 0},
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 1: Discovery
        if discovery:
            new_episode_ids = self.run_discovery()
            results["discovery"]["new_episodes"] = len(new_episode_ids)
        
        # Step 2-4: Processing
        if processing:
            processing_results = self.run_processing(episode_limit)
            results["processing"] = processing_results
        
        # Step 5: Scheduler (runs in background)
        self.run_scheduler()
        
        logger.info("🎉 Pipeline execution complete!")
        logger.info(f"📊 Results: {results}")
        
        return results
    
    def _select_formats_for_insight(self, insight) -> List[str]:
        """Select tweet formats based on insight type and content strategy."""
        
        # Strategy: Mix formats based on insight type
        format_map = {
            "powerful_quote": ["single", "quote_card"],
            "surprising_fact": ["single", "thread"],
            "hot_take": ["single", "poll"],
            "key_concept": ["thread"],
            "actionable_advice": ["thread", "single"],
            "funny_moment": ["single"],
            "personal_story": ["thread", "single"]
        }
        
        suggested_formats = format_map.get(insight.insight_type.value, ["single"])
        
        # Apply content mix ratios
        content_mix = self.config["content_mix"]
        
        # For now, return the first suggested format
        # In a more sophisticated version, this would balance the content mix
        return suggested_formats[:1]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        episode_stats = self.ingestor.db.get_pending_episodes(1000)  # Get all for counting
        tweet_stats = self.scheduler.queue.get_queue_stats()
        
        return {
            "episodes": {
                "total_discovered": len(episode_stats),
                "pending_processing": len([ep for ep in episode_stats if ep.processing_status == "pending"]),
                "completed": len([ep for ep in episode_stats if ep.processing_status == "completed"])
            },
            "tweets": tweet_stats,
            "performance": self.scheduler.get_performance_stats(),
            "last_updated": datetime.now().isoformat()
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Podcasts TLDR Viral Content Machine")
    parser.add_argument("--config", default="viral_config.json", help="Configuration file")
    parser.add_argument("--mode", choices=["discovery", "processing", "scheduler", "web", "full"], 
                       default="full", help="Pipeline mode to run")
    parser.add_argument("--episode-limit", type=int, default=5, help="Max episodes to process")
    parser.add_argument("--web-port", type=int, default=5000, help="Web interface port")
    parser.add_argument("--stats", action="store_true", help="Show pipeline statistics")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = PodcastsTLDRPipeline(args.config)
        
        if args.stats:
            # Show statistics
            stats = pipeline.get_stats()
            print(json.dumps(stats, indent=2))
            return
        
        # Run based on mode
        if args.mode == "discovery":
            pipeline.run_discovery()
        
        elif args.mode == "processing":
            pipeline.run_processing(args.episode_limit)
        
        elif args.mode == "scheduler":
            pipeline.run_scheduler()
            # Keep running
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                pipeline.scheduler.stop_scheduler()
        
        elif args.mode == "web":
            pipeline.run_web_interface(args.web_port)
        
        elif args.mode == "full":
            pipeline.run_full_pipeline(episode_limit=args.episode_limit)
        
        logger.info("🎯 Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()