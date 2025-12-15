#!/usr/bin/env python3
"""
Main orchestrator for the Podcasts TLDR Viral Content Machine.
The complete pipeline from podcast ingestion to viral tweet posting.
"""

import logging
import sys
import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Rich library for beautiful terminal UI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich import box

# Load environment variables
load_dotenv()

# Import all viral content components
from src.podcast_ingestor import PodcastIngestor
from src.viral_transcriber import ViralTranscriber
from src.viral_insight_extractor import ViralContentAnalyzer, InsightDatabase
from src.viral_tweet_crafter import ViralTweetCrafter
from src.viral_scheduler import ViralScheduler, PostingStrategy
from src.web_interface import create_app, create_templates
from src.hashtag_optimizer import HashtagOptimizer
from src.error_handling import (
    ErrorAggregator, ErrorClassifier, ErrorCategory,
    retry_with_backoff, safe_execute, try_or_default,
    get_circuit_breaker, ResilientPipeline
)

# Initialize Rich console
console = Console()

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
        # Show beautiful ASCII art header
        self._show_header()

        # Initialize error tracking
        self.error_aggregator = ErrorAggregator()
        self.checkpoint_file = "data/pipeline_checkpoint.json"

        with console.status("[bold cyan]Initializing pipeline...", spinner="dots"):
            self.config = self._load_config(config_file)

            # Initialize data directories
            Path("data").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            Path("cache").mkdir(exist_ok=True)
            Path("output").mkdir(exist_ok=True)

            # Initialize components with error recovery
            self._init_components_safe()

        console.print("✅ [bold green]Podcasts TLDR Pipeline initialized successfully![/bold green]")

    def _show_header(self):
        """Display beautiful ASCII art header."""
        header_art = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🎙️  PODCASTS TLDR → X.COM THREADS PIPELINE  🚀             ║
║                                                               ║
║   Transform podcast episodes into viral educational threads   ║
║   Local transcription • AI insights • Auto scheduling        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """

        console.print(Panel(
            Text(header_art, style="bold cyan"),
            border_style="bright_cyan",
            box=box.DOUBLE
        ))
        console.print()
    
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
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)

        # Load Twitter credentials from environment if not in config
        if default_config.get("x_accounts", {}).get("podcasts_tldr", {}).get("consumer_key") == "":
            default_config["x_accounts"]["podcasts_tldr"] = {
                "consumer_key": os.getenv("MAIN_API_KEY", ""),
                "consumer_secret": os.getenv("MAIN_API_SECRET", ""),
                "access_token": os.getenv("MAIN_ACCESS_TOKEN", ""),
                "access_token_secret": os.getenv("MAIN_ACCESS_TOKEN_SECRET", ""),
                "bearer_token": os.getenv("MAIN_BEARER_TOKEN", "")
            }
            logger.info("✅ Loaded Twitter API credentials from environment variables")

        return default_config
    
    def _init_components_safe(self):
        """Initialize all pipeline components with error recovery."""
        try:
            self._init_components()
        except ValueError as e:
            # API key missing - provide helpful guidance
            console.print(f"\n[bold red]Configuration Error:[/bold red] {e}")
            console.print("\n[yellow]To fix this, set one of these environment variables:[/yellow]")
            console.print("  • OPENAI_API_KEY=your_openai_key")
            console.print("  • OPENROUTER_API_KEY=your_openrouter_key (with USE_OPENROUTER=true)")
            console.print("\n[dim]You can add these to your .env file[/dim]")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"\n[bold red]Failed to initialize pipeline:[/bold red] {e}")
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            raise SystemExit(1)

    def _init_components(self):
        """Initialize all pipeline components."""
        
        # 1. Podcast Ingestor
        self.ingestor = PodcastIngestor(
            feed_urls=self.config["podcast_feeds"],
            max_episodes_per_feed=20
        )

        # Backfill missing episode numbers from titles and RSS feeds
        backfilled = self.ingestor.db.backfill_episode_numbers(
            feed_urls=self.config["podcast_feeds"]
        )
        if backfilled > 0:
            logger.info(f"Backfilled episode numbers for {backfilled} existing episodes")

        # Configure OpenAI client to use OpenRouter if enabled
        use_openrouter = os.getenv("USE_OPENROUTER", "").lower() == "true"

        if use_openrouter:
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            # Configure OpenAI client globally to use OpenRouter endpoint
            openai_base_url = "https://openrouter.ai/api/v1"
            logger.info("Using OpenRouter for AI models")
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            openai_base_url = None
            logger.info("Using direct OpenAI API")

        if not api_key:
            logger.error("No API key found in environment variables!")
            raise ValueError("API key required (OPENROUTER_API_KEY or OPENAI_API_KEY)")

        # Store config for components
        self.api_key = api_key
        self.openai_base_url = openai_base_url

        # 2. Viral Transcriber
        self.transcriber = ViralTranscriber(
            openai_api_key=api_key,
            preferred_methods=self.config["transcription_methods"],
            enable_speaker_id=True,
            enable_viral_detection=True
        )

        # 3. Insight Extractor
        self.insight_analyzer = ViralContentAnalyzer(
            openai_api_key=api_key,
            model="gpt-4-turbo-preview" if use_openrouter else "gpt-4",
            base_url=openai_base_url
        )

        self.insight_db = InsightDatabase()

        # 4. Tweet Crafter
        self.tweet_crafter = ViralTweetCrafter(
            openai_api_key=api_key,
            model="gpt-4-turbo-preview" if use_openrouter else "gpt-4",
            base_url=openai_base_url
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
        console.print("\n[bold cyan]🔍 STEP 1: Podcast Discovery[/bold cyan]")
        console.print("─" * 60)

        with console.status("[bold yellow]Scanning podcast feeds...", spinner="dots"):
            new_episodes = self.ingestor.discover_new_episodes()

        if not new_episodes:
            console.print("[yellow]ℹ️  No new episodes found[/yellow]")
            return []

        # Display discoveries in a table
        table = Table(title="📻 New Episodes Discovered", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Podcast", style="cyan", no_wrap=False, width=25)
        table.add_column("Episode", style="white", no_wrap=False, width=35)
        table.add_column("Viral Score", justify="center", style="green")

        for episode in new_episodes:
            score_color = "green" if episode.viral_score >= 0.6 else "yellow" if episode.viral_score >= 0.4 else "red"
            table.add_row(
                episode.podcast_name,
                episode.title[:60] + "..." if len(episode.title) > 60 else episode.title,
                f"[{score_color}]{episode.viral_score:.2f}[/{score_color}]"
            )

        console.print(table)
        console.print(f"[bold green]✅ Found {len(new_episodes)} new episode(s)[/bold green]\n")

        return [ep.episode_id for ep in new_episodes]
    
    def run_processing(self, episode_limit: int = 5) -> Dict[str, Any]:
        """
        Run complete processing pipeline for pending episodes.
        Steps 2-4: Transcribe → Extract insights → Craft tweets.
        
        Features:
        - Checkpoint support for resuming after crashes
        - Per-episode error isolation (one failure doesn't stop others)
        - Error aggregation and reporting
        - Graceful degradation
        """
        console.print("\n[bold cyan]⚡ STEP 2-4: Episode Processing Pipeline[/bold cyan]")
        console.print("─" * 60)

        # Load checkpoint for resume support
        checkpoint = self._load_checkpoint()
        completed_episodes = set(checkpoint.get("completed_episodes", []))
        
        if completed_episodes:
            console.print(f"[dim]  Resuming from checkpoint ({len(completed_episodes)} episodes already done)[/dim]")

        # Get episodes for processing
        with console.status("[bold yellow]Loading pending episodes...", spinner="dots"):
            pending_episodes = self.ingestor.get_episodes_for_processing(episode_limit)

        if not pending_episodes:
            console.print("[yellow]ℹ️  No episodes pending processing[/yellow]")
            return {"processed": 0, "tweets_created": 0}

        # Filter out already-completed episodes (from checkpoint)
        episodes_to_process = [
            ep for ep in pending_episodes 
            if ep.episode_id not in completed_episodes
        ]
        
        if len(episodes_to_process) < len(pending_episodes):
            skipped = len(pending_episodes) - len(episodes_to_process)
            console.print(f"[dim]  Skipping {skipped} already-processed episode(s)[/dim]")

        if not episodes_to_process:
            console.print("[yellow]ℹ️  All pending episodes already processed[/yellow]")
            return {"processed": 0, "tweets_created": 0, "skipped": len(pending_episodes)}

        console.print(f"[bold white]Processing {len(episodes_to_process)} episode(s)...[/bold white]\n")

        total_tweets = 0
        processed_count = 0
        failed_count = 0
        episode_errors: Dict[str, str] = {}

        # Create overall progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as overall_progress:

            episodes_task = overall_progress.add_task(
                "[cyan]Processing episodes",
                total=len(episodes_to_process)
            )

            for episode in episodes_to_process:
                try:
                    # Display current episode
                    console.print(f"\n[bold white]🎙️  {episode.podcast_name}[/bold white]")
                    console.print(f"[dim]{episode.title[:80]}...[/dim]\n")

                    # Mark as processing
                    self.ingestor.mark_episode_processing(episode.episode_id)

                    # Step 2: Transcribe with viral enhancements
                    step_task = overall_progress.add_task(
                        f"[yellow]  → Transcribing audio",
                        total=100
                    )

                    try:
                        overall_progress.update(step_task, advance=10)
                        transcription = self.transcriber.transcribe_for_viral_content(
                            audio_url=episode.audio_url,
                            youtube_urls=episode.youtube_urls or [],
                            title=episode.title,
                            language=None
                        )
                        overall_progress.update(step_task, advance=90)

                        if not transcription or not transcription.text:
                            overall_progress.remove_task(step_task)
                            console.print("[red]❌ Transcription failed - empty result[/red]")
                            self.ingestor.mark_episode_failed(episode.episode_id, "Transcription returned empty result")
                            overall_progress.update(episodes_task, advance=1)
                            continue

                    except Exception as transcription_error:
                        overall_progress.remove_task(step_task)
                        console.print(f"[red]❌ Transcription error: {str(transcription_error)[:60]}...[/red]")
                        self.ingestor.mark_episode_failed(episode.episode_id, f"Transcription error: {str(transcription_error)}")
                        overall_progress.update(episodes_task, advance=1)
                        continue

                    overall_progress.remove_task(step_task)
                    console.print(f"[green]  ✓ Transcribed {len(transcription.segments)} segments[/green]")

                    # Step 3: Extract key educational points
                    step_task = overall_progress.add_task(
                        f"[yellow]  → Extracting key points",
                        total=100
                    )

                    overall_progress.update(step_task, advance=20)
                    key_points = self.insight_analyzer.extract_key_points(
                        transcription=transcription,
                        podcast_name=episode.podcast_name,
                        episode_title=episode.title,
                        num_points=self.config.get("key_points_per_episode", 5)
                    )
                    overall_progress.update(step_task, advance=80)

                    if not key_points or len(key_points) < 5:
                        overall_progress.remove_task(step_task)
                        console.print("[red]❌ Could not extract 5 key points[/red]")
                        self.ingestor.mark_episode_completed(episode.episode_id, insights_count=0)
                        overall_progress.update(episodes_task, advance=1)
                        continue

                    overall_progress.remove_task(step_task)
                    console.print(f"[green]  ✓ Extracted {len(key_points)} educational points[/green]")

                    # Save insights to database
                    self.insight_db.save_insights(key_points, episode.episode_id)

                    # Step 3.5: Download thumbnail using multi-source fetcher
                    thumbnail_path = None
                    step_task = overall_progress.add_task(
                        f"[yellow]  → Fetching thumbnail",
                        total=100
                    )

                    try:
                        from src.thumbnail_fetcher import ThumbnailFetcher

                        # Use shared thumbnail fetcher (created once)
                        if not hasattr(self, '_thumbnail_fetcher'):
                            self._thumbnail_fetcher = ThumbnailFetcher()

                        thumbnail_path = self._thumbnail_fetcher.get_thumbnail(episode)
                        overall_progress.update(step_task, advance=100)

                        if thumbnail_path:
                            console.print(f"[green]  ✓ Thumbnail acquired[/green]")
                            # Update database with thumbnail path
                            self.ingestor.db.update_thumbnail_path(episode.episode_id, thumbnail_path)
                        else:
                            console.print(f"[yellow]  ⚠ No thumbnail available[/yellow]")

                    except Exception as e:
                        console.print(f"[yellow]  ⚠ Thumbnail error: {str(e)[:40]}...[/yellow]")
                        logger.warning(f"Thumbnail fetch failed: {e}")
                    finally:
                        # Always remove the task, whether success or failure
                        try:
                            overall_progress.remove_task(step_task)
                        except KeyError:
                            pass  # Task already removed

                    # Step 4: Get Twitter handles for episode
                    step_task = overall_progress.add_task(
                        f"[yellow]  → Finding Twitter handles",
                        total=100
                    )

                    from src.twitter_handle_finder import TwitterHandleFinder

                    handle_finder = TwitterHandleFinder(self.config.get("podcast_handles", {}))
                    podcast_handle, host_handles, guest_handles = handle_finder.get_all_handles_for_episode(
                        podcast_name=episode.podcast_name,
                        episode_title=episode.title,
                        episode_description=episode.description
                    )
                    overall_progress.update(step_task, advance=100)

                    overall_progress.remove_task(step_task)
                    # Filter out None values before joining
                    handles_found = [h for h in ([podcast_handle] + (host_handles or []) + (guest_handles or [])) if h is not None]
                    if handles_found:
                        console.print(f"[green]  ✓ Found {len(handles_found)} handle(s): {', '.join(handles_found[:3])}...[/green]")
                    else:
                        console.print("[yellow]  ℹ No Twitter handles found for this podcast[/yellow]")

                    # Step 5: Create ONE educational thread per episode (6 tweets)
                    step_task = overall_progress.add_task(
                        f"[yellow]  → Generating thread",
                        total=100
                    )

                    overall_progress.update(step_task, advance=20)
                    thread = self.tweet_crafter.thread_builder.build_educational_thread(
                        key_points=key_points,
                        episode_title=episode.title,
                        podcast_name=episode.podcast_name,
                        podcast_handle=podcast_handle,
                        host_handles=host_handles,
                        guest_handles=guest_handles,
                        episode_number=episode.episode_number
                    )
                    overall_progress.update(step_task, advance=80)

                    if not thread or len(thread) != 6:
                        overall_progress.remove_task(step_task)
                        console.print("[red]❌ Failed to create valid 6-tweet thread[/red]")
                        self.ingestor.mark_episode_failed(episode.episode_id, "Thread generation failed")
                        overall_progress.update(episodes_task, advance=1)
                        continue

                    overall_progress.remove_task(step_task)
                    console.print(f"[green]  ✓ Generated 6-tweet educational thread[/green]")

                    # Step 6: Validate thread
                    step_task = overall_progress.add_task(
                        f"[yellow]  → Validating thread",
                        total=100
                    )

                    from src.content_validator import ContentValidator
                    validator = ContentValidator()
                    is_valid, errors, cleaned_thread = validator.validate_thread(thread)
                    overall_progress.update(step_task, advance=100)

                    # Use cleaned thread (filler words auto-stripped)
                    thread = cleaned_thread

                    if not is_valid:
                        overall_progress.remove_task(step_task)
                        console.print("[red]❌ Thread validation failed:[/red]")
                        for error in errors:
                            console.print(f"[red]     • {error}[/red]")
                        self.ingestor.mark_episode_failed(episode.episode_id, f"Validation failed: {errors[0]}")
                        overall_progress.update(episodes_task, advance=1)
                        continue

                    overall_progress.remove_task(step_task)
                    console.print(f"[green]  ✓ Thread validated successfully[/green]")

                    # Step 6b: Add strategic hashtags to final tweet
                    try:
                        hashtag_optimizer = HashtagOptimizer()
                        # Get content-based hashtags
                        hashtags = hashtag_optimizer.get_hashtags_for_content(
                            content=' '.join(thread[:3]),  # Analyze first 3 tweets for context
                            podcast_name=episode.podcast_name,
                            episode_title=episode.title,
                            max_hashtags=3
                        )
                        if hashtags and thread:
                            # Add hashtags to the final tweet (tweet 6 - CTA tweet)
                            final_tweet = thread[-1]
                            hashtag_str = ' '.join(hashtags)
                            # Only add if it fits within 280 chars
                            if len(final_tweet) + len(hashtag_str) + 2 <= 280:
                                thread[-1] = f"{final_tweet}\n\n{hashtag_str}"
                                console.print(f"[green]  ✓ Added hashtags: {hashtag_str}[/green]")
                            else:
                                console.print(f"[yellow]  ℹ Skipped hashtags (tweet too long)[/yellow]")
                    except Exception as e:
                        console.print(f"[yellow]  ℹ Could not add hashtags: {e}[/yellow]")

                    # Step 7: Schedule the thread using new thread-based storage
                    step_task = overall_progress.add_task(
                        f"[yellow]  → Scheduling thread",
                        total=100
                    )

                    for account_name in self.config["x_accounts"].keys():
                        # Use new thread scheduling method for proper thread posting
                        thread_id = self.scheduler.schedule_thread(
                            thread_tweets=thread,  # List of 6 tweet strings
                            account_name=account_name,
                            podcast_name=episode.podcast_name,
                            podcast_handle=podcast_handle,
                            host_handles=host_handles,
                            guest_handles=guest_handles,
                            episode_id=episode.episode_id,
                            episode_title=episode.title,
                            thumbnail_path=thumbnail_path,
                            scheduled_time=None  # Will use default (1 hour from now)
                        )

                        if thread_id:
                            total_tweets += 6  # Count all 6 tweets in thread

                    overall_progress.update(step_task, advance=100)
                    overall_progress.remove_task(step_task)
                    console.print(f"[green]  ✓ Scheduled 6-tweet thread[/green]")

                    # Mark episode as completed
                    self.ingestor.mark_episode_completed(
                        episode.episode_id,
                        insights_count=len(key_points),
                        tweets_generated=6  # Always 6 tweets per episode
                    )

                    processed_count += 1
                    console.print(f"\n[bold green]✅ Episode completed successfully![/bold green]")

                    # Save checkpoint after each successful episode
                    self._save_checkpoint_episode(episode.episode_id)

                    # Update overall progress
                    overall_progress.update(episodes_task, advance=1)

                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)[:200]
                    episode_errors[episode.episode_id] = error_msg
                    
                    # Classify the error to decide how to handle it
                    error_category = ErrorClassifier.classify(e)
                    
                    if error_category == ErrorCategory.RESOURCE:
                        # Resource error - pause and retry later
                        console.print(f"\n[bold yellow]⚠️ Resource issue detected, pausing...[/bold yellow]")
                        console.print(f"[dim]  Error: {error_msg}[/dim]")
                        logger.warning(f"Resource error processing {episode.title}: {e}")
                        time.sleep(30)  # Wait before continuing
                    elif error_category == ErrorCategory.TRANSIENT:
                        # Transient error - log but continue
                        console.print(f"\n[yellow]⚠️ Temporary error, skipping episode: {error_msg[:60]}...[/yellow]")
                        logger.warning(f"Transient error processing {episode.title}: {e}")
                    else:
                        # Permanent or unknown error
                        console.print(f"\n[bold red]❌ Error processing episode: {error_msg[:80]}...[/bold red]")
                        logger.error(f"Error processing {episode.title}: {e}", exc_info=True)
                    
                    # Record error in aggregator
                    self.error_aggregator.record(e, context={
                        "episode_id": episode.episode_id,
                        "episode_title": episode.title,
                        "podcast_name": episode.podcast_name
                    })
                    
                    self.ingestor.mark_episode_failed(episode.episode_id, error_msg)
                    overall_progress.update(episodes_task, advance=1)
                    continue

        # Summary with error reporting
        console.print("\n" + "─" * 60)
        console.print(f"[bold green]🎯 Processing Complete![/bold green]")
        console.print(f"[white]   • Episodes processed: {processed_count}[/white]")
        console.print(f"[white]   • Tweets scheduled: {total_tweets}[/white]")
        
        if failed_count > 0:
            console.print(f"[yellow]   • Episodes failed: {failed_count}[/yellow]")
            
            # Show error summary
            error_summary = self.error_aggregator.get_summary()
            if error_summary.get("by_category"):
                console.print(f"[dim]   • Error breakdown: {error_summary['by_category']}[/dim]")
        
        console.print()

        # Clear checkpoint on successful completion
        if failed_count == 0:
            self._clear_checkpoint()

        return {
            "processed": processed_count,
            "tweets_created": total_tweets,
            "failed": failed_count,
            "errors": episode_errors,
            "episodes": [ep.episode_id for ep in episodes_to_process[:processed_count]]
        }
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint for resume support."""
        try:
            checkpoint_path = Path(self.checkpoint_file)
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
        return {}

    def _save_checkpoint_episode(self, episode_id: str):
        """Save checkpoint after processing an episode."""
        try:
            checkpoint = self._load_checkpoint()
            if "completed_episodes" not in checkpoint:
                checkpoint["completed_episodes"] = []
            
            if episode_id not in checkpoint["completed_episodes"]:
                checkpoint["completed_episodes"].append(episode_id)
            
            checkpoint["last_update"] = datetime.now().isoformat()
            
            checkpoint_path = Path(self.checkpoint_file)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _clear_checkpoint(self):
        """Clear checkpoint after successful completion."""
        try:
            checkpoint_path = Path(self.checkpoint_file)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Cleared pipeline checkpoint")
        except Exception as e:
            logger.warning(f"Failed to clear checkpoint: {e}")

    def run_scheduler(self):
        """
        Start the automated scheduler.
        Step 5: Post tweets at optimal times.
        """
        console.print("\n[bold cyan]🚀 STEP 5: Tweet Scheduler[/bold cyan]")
        console.print("─" * 60)

        with console.status("[bold yellow]Starting scheduler...", spinner="dots"):
            self.scheduler.start_scheduler()

        console.print("[bold green]✅ Scheduler running in background[/bold green]\n")
    
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
        start_time = datetime.now()

        results = {
            "discovery": {"new_episodes": 0},
            "processing": {"processed": 0, "tweets_created": 0},
            "timestamp": start_time.isoformat()
        }

        # Step 1: Discovery
        if discovery:
            new_episode_ids = self.run_discovery()
            results["discovery"]["new_episodes"] = len(new_episode_ids)

        # Step 1.5: Auto-backfill missing thumbnails
        try:
            from src.thumbnail_fetcher import backfill_thumbnails, ThumbnailFetcher
            console.print("\n[bold cyan]📷 Backfilling Missing Thumbnails[/bold cyan]")

            if not hasattr(self, '_thumbnail_fetcher'):
                self._thumbnail_fetcher = ThumbnailFetcher()

            backfill_result = backfill_thumbnails(
                db=self.ingestor.db,
                fetcher=self._thumbnail_fetcher,
                limit=20,
                delay=0.5
            )
            results["thumbnails"] = backfill_result
        except Exception as e:
            logger.warning(f"Thumbnail backfill failed: {e}")
            results["thumbnails"] = {"error": str(e)}

        # Step 2-4: Processing
        if processing:
            processing_results = self.run_processing(episode_limit)
            results["processing"] = processing_results

        # Step 5: Scheduler (runs in background)
        self.run_scheduler()

        # Display final summary
        elapsed = (datetime.now() - start_time).total_seconds()

        console.print("\n" + "═" * 60)
        summary_table = Table(title="🎉 Pipeline Execution Complete!", box=box.DOUBLE_EDGE, show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan", width=30)
        summary_table.add_column("Value", style="green", justify="right", width=20)

        summary_table.add_row("New Episodes Discovered", str(results["discovery"]["new_episodes"]))
        summary_table.add_row("Episodes Processed", str(results["processing"]["processed"]))
        summary_table.add_row("Tweets Scheduled", str(results["processing"]["tweets_created"]))
        summary_table.add_row("Execution Time", f"{elapsed:.1f}s")

        console.print(summary_table)
        console.print("═" * 60 + "\n")

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

        stats = {
            "episodes": {
                "total_discovered": len(episode_stats),
                "pending_processing": len([ep for ep in episode_stats if ep.processing_status == "pending"]),
                "completed": len([ep for ep in episode_stats if ep.processing_status == "completed"])
            },
            "tweets": tweet_stats,
            "performance": self.scheduler.get_performance_stats(),
            "last_updated": datetime.now().isoformat()
        }

        # Display stats beautifully with rich
        console.print("\n" + "═" * 60)
        console.print(Panel.fit(
            "[bold cyan]📊 PIPELINE STATISTICS[/bold cyan]",
            border_style="cyan"
        ))

        # Episodes table
        episodes_table = Table(title="Episodes", box=box.ROUNDED, show_header=True, header_style="bold yellow")
        episodes_table.add_column("Status", style="cyan")
        episodes_table.add_column("Count", justify="right", style="white")

        episodes_table.add_row("Total Discovered", str(stats["episodes"]["total_discovered"]))
        episodes_table.add_row("Pending Processing", f"[yellow]{stats['episodes']['pending_processing']}[/yellow]")
        episodes_table.add_row("Completed", f"[green]{stats['episodes']['completed']}[/green]")

        console.print(episodes_table)

        # Tweets table
        tweets_table = Table(title="Tweets", box=box.ROUNDED, show_header=True, header_style="bold yellow")
        tweets_table.add_column("Metric", style="cyan")
        tweets_table.add_column("Value", justify="right", style="white")

        for key, value in tweet_stats.items():
            tweets_table.add_row(key.replace("_", " ").title(), str(value))

        console.print(tweets_table)
        console.print("═" * 60 + "\n")

        return stats


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
            pipeline.get_stats()
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
                console.print("[yellow]Press Ctrl+C to stop the scheduler[/yellow]")
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping scheduler...[/yellow]")
                pipeline.scheduler.stop_scheduler()
                console.print("[green]✅ Scheduler stopped[/green]")

        elif args.mode == "web":
            console.print(f"\n[bold cyan]🌐 Starting web interface on port {args.web_port}...[/bold cyan]")
            pipeline.run_web_interface(args.web_port)

        elif args.mode == "full":
            pipeline.run_full_pipeline(episode_limit=args.episode_limit)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Pipeline interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]❌ PIPELINE FAILED[/bold red]")
        console.print(Panel(
            f"[red]{str(e)}[/red]",
            title="Error Details",
            border_style="red"
        ))
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()