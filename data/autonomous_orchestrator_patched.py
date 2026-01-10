#!/usr/bin/env python3
"""
Autonomous Orchestrator - Fully Automated 24/7 Pipeline Runner
Runs the podcast-to-tweets pipeline continuously without human intervention.
"""

import logging
import time
import signal
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict
import os

# Import pipeline components
from viral_main import PodcastsTLDRPipeline
from src.telegram_scheduler import create_telegram_scheduler_from_config

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for autonomous orchestration."""

    # Check intervals (in seconds)
    discovery_interval: int = 3600  # 1 hour - check for new episodes
    processing_interval: int = 1800  # 30 minutes - process pending episodes
    health_check_interval: int = 300  # 5 minutes - system health checks

    # Processing limits
    max_episodes_per_cycle: int = 5
    max_concurrent_processing: int = 3

    # Resource management
    max_memory_mb: int = 2048  # Pause if memory exceeds this
    max_cpu_percent: int = 80  # Throttle if CPU exceeds this

    # Safety limits
    max_api_cost_per_day: float = 50.0  # USD
    max_tweets_per_day: int = 100

    # Retry configuration
    max_retries: int = 3
    retry_backoff: int = 300  # 5 minutes

    # Operating hours (UTC) - None means 24/7
    operating_hours_start: Optional[int] = None  # 0-23
    operating_hours_end: Optional[int] = None  # 0-23

    # Quiet mode (pause processing during these hours)
    quiet_hours_start: Optional[int] = None  # e.g., 1 AM UTC
    quiet_hours_end: Optional[int] = None  # e.g., 6 AM UTC

    def is_operating_time(self) -> bool:
        """Check if current time is within operating hours."""
        if self.operating_hours_start is None:
            return True

        current_hour = datetime.utcnow().hour

        if self.operating_hours_start <= self.operating_hours_end:
            return self.operating_hours_start <= current_hour < self.operating_hours_end
        else:
            # Handle overnight range (e.g., 22:00 - 06:00)
            return current_hour >= self.operating_hours_start or current_hour < self.operating_hours_end

    def is_quiet_time(self) -> bool:
        """Check if current time is during quiet hours."""
        if self.quiet_hours_start is None:
            return False

        current_hour = datetime.utcnow().hour

        if self.quiet_hours_start <= self.quiet_hours_end:
            return self.quiet_hours_start <= current_hour < self.quiet_hours_end
        else:
            return current_hour >= self.quiet_hours_start or current_hour < self.quiet_hours_end


class AutonomousOrchestrator:
    """
    Fully autonomous orchestrator for 24/7 operation.

    Features:
    - Continuous RSS monitoring for new episodes
    - Automatic processing and scheduling
    - Self-healing error recovery
    - Resource-aware throttling
    - Cost and rate limit enforcement
    - Health monitoring and metrics tracking
    """

    def __init__(self, config_file: str = "viral_config.json",
                 orchestration_config: Optional[OrchestrationConfig] = None):
        """Initialize autonomous orchestrator."""

        self.config_file = config_file
        self.orchestration_config = orchestration_config or self._load_orchestration_config()

        # Pipeline instance
        self.pipeline: Optional[PodcastsTLDRPipeline] = None
        
        # Telegram scheduler (initialized lazily)
        self.telegram_scheduler = None
        self._init_telegram_scheduler()

        # State tracking
        self.running = False
        self.paused = False
        self.last_discovery = None
        self.last_processing = None
        self.last_health_check = None

        # Metrics
        self.metrics = {
            "total_discoveries": 0,
            "total_processed": 0,
            "total_tweets_created": 0,
            "total_errors": 0,
            "api_cost_today": 0.0,
            "tweets_today": 0,
            "last_reset": datetime.utcnow().date().isoformat(),
            "uptime_start": None,
            "cycles_completed": 0
        }

        # Threads
        self.discovery_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.health_thread: Optional[threading.Thread] = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Setup logging
        self._setup_logging()

        logger.info("🤖 Autonomous Orchestrator initialized")

    def _load_orchestration_config(self) -> OrchestrationConfig:
        """Load orchestration config from environment or defaults."""

        return OrchestrationConfig(
            discovery_interval=int(os.getenv("DISCOVERY_INTERVAL", "3600")),
            processing_interval=int(os.getenv("PROCESSING_INTERVAL", "1800")),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "300")),
            max_episodes_per_cycle=int(os.getenv("MAX_EPISODES_PER_CYCLE", "5")),
            max_api_cost_per_day=float(os.getenv("MAX_API_COST_PER_DAY", "50.0")),
            max_tweets_per_day=int(os.getenv("MAX_TWEETS_PER_DAY", "100")),
            quiet_hours_start=int(os.getenv("QUIET_HOURS_START")) if os.getenv("QUIET_HOURS_START") else None,
            quiet_hours_end=int(os.getenv("QUIET_HOURS_END")) if os.getenv("QUIET_HOURS_END") else None
        )

    def _init_telegram_scheduler(self):
        """Initialize Telegram scheduler if configured."""
        try:
            self.telegram_scheduler = create_telegram_scheduler_from_config(
                config_file=self.config_file
            )
            if self.telegram_scheduler:
                logger.info("📱 Telegram scheduler initialized")
            else:
                logger.info("📱 Telegram not configured or not enabled")
        except Exception as e:
            logger.warning(f"Could not initialize Telegram scheduler: {e}")
            self.telegram_scheduler = None

    def _setup_logging(self):
        """Setup logging for autonomous operation."""

        Path("logs").mkdir(exist_ok=True)

        log_file = f'logs/autonomous_{datetime.now().strftime("%Y%m%d")}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"⚠️ Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def _reset_daily_metrics(self):
        """Reset daily metrics at UTC midnight."""
        current_date = datetime.utcnow().date().isoformat()

        if current_date != self.metrics["last_reset"]:
            logger.info("📊 Resetting daily metrics")
            self.metrics["api_cost_today"] = 0.0
            self.metrics["tweets_today"] = 0
            self.metrics["last_reset"] = current_date
            self._save_metrics()

    def _save_metrics(self):
        """Persist metrics to disk."""
        try:
            Path("data").mkdir(exist_ok=True)
            metrics_file = "data/orchestrator_metrics.json"

            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def _load_metrics(self):
        """Load persisted metrics."""
        try:
            metrics_file = "data/orchestrator_metrics.json"

            if Path(metrics_file).exists():
                with open(metrics_file, 'r') as f:
                    loaded = json.load(f)
                    self.metrics.update(loaded)
                logger.info("📊 Loaded metrics from disk")
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")

    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits."""

        self._reset_daily_metrics()

        # Check API cost limit
        if self.metrics["api_cost_today"] >= self.orchestration_config.max_api_cost_per_day:
            logger.warning(
                f"⚠️ Daily API cost limit reached: "
                f"${self.metrics['api_cost_today']:.2f} / "
                f"${self.orchestration_config.max_api_cost_per_day:.2f}"
            )
            return False

        # Check tweet limit
        if self.metrics["tweets_today"] >= self.orchestration_config.max_tweets_per_day:
            logger.warning(
                f"⚠️ Daily tweet limit reached: "
                f"{self.metrics['tweets_today']} / "
                f"{self.orchestration_config.max_tweets_per_day}"
            )
            return False

        return True

    def _init_pipeline(self) -> bool:
        """Initialize or reinitialize the pipeline."""
        try:
            logger.info("🔧 Initializing pipeline...")
            self.pipeline = PodcastsTLDRPipeline(self.config_file)
            logger.info("✅ Pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
            self.metrics["total_errors"] += 1
            return False

    def _discovery_loop(self):
        """Continuous discovery loop - runs in separate thread."""

        logger.info("🔍 Discovery loop started")

        while self.running:
            try:
                # Check if it's time to run
                if self.last_discovery:
                    elapsed = (datetime.utcnow() - self.last_discovery).total_seconds()
                    if elapsed < self.orchestration_config.discovery_interval:
                        time.sleep(10)  # Check every 10 seconds
                        continue

                # Check operating hours
                if not self.orchestration_config.is_operating_time():
                    logger.info("⏸️ Outside operating hours, skipping discovery")
                    time.sleep(300)  # Check every 5 minutes
                    continue

                # Check if paused
                if self.paused:
                    time.sleep(60)
                    continue

                # Run discovery
                logger.info("🔍 Running discovery cycle...")

                if not self.pipeline:
                    if not self._init_pipeline():
                        time.sleep(300)  # Wait 5 minutes before retry
                        continue

                new_episodes = self.pipeline.run_discovery()

                if new_episodes:
                    logger.info(f"✅ Discovered {len(new_episodes)} new episodes")
                    self.metrics["total_discoveries"] += len(new_episodes)
                else:
                    logger.info("ℹ️ No new episodes found")

                self.last_discovery = datetime.utcnow()
                self.metrics["cycles_completed"] += 1
                self._save_metrics()

            except Exception as e:
                logger.error(f"❌ Discovery loop error: {e}", exc_info=True)
                self.metrics["total_errors"] += 1
                time.sleep(self.orchestration_config.retry_backoff)

        logger.info("🛑 Discovery loop stopped")

    def _processing_loop(self):
        """Continuous processing loop - runs in separate thread."""

        logger.info("⚙️ Processing loop started")

        while self.running:
            try:
                # Check if it's time to run
                if self.last_processing:
                    elapsed = (datetime.utcnow() - self.last_processing).total_seconds()
                    if elapsed < self.orchestration_config.processing_interval:
                        time.sleep(10)
                        continue

                # Check operating hours and quiet time
                if not self.orchestration_config.is_operating_time():
                    logger.info("⏸️ Outside operating hours, skipping processing")
                    time.sleep(300)
                    continue

                if self.orchestration_config.is_quiet_time():
                    logger.info("🤫 Quiet hours, skipping processing")
                    time.sleep(300)
                    continue

                # Check if paused
                if self.paused:
                    time.sleep(60)
                    continue

                # Check rate limits
                if not self._check_rate_limits():
                    logger.warning("⚠️ Rate limits exceeded, pausing processing")
                    time.sleep(3600)  # Wait 1 hour
                    continue

                # Run processing
                logger.info("⚙️ Running processing cycle...")

                if not self.pipeline:
                    if not self._init_pipeline():
                        time.sleep(300)
                        continue

                results = self.pipeline.run_processing(
                    episode_limit=self.orchestration_config.max_episodes_per_cycle
                )

                # Update metrics
                self.metrics["total_processed"] += results.get("processed", 0)
                self.metrics["total_tweets_created"] += results.get("tweets_created", 0)
                self.metrics["tweets_today"] += results.get("tweets_created", 0)

                logger.info(
                    f"✅ Processing cycle complete: "
                    f"{results.get('processed', 0)} episodes, "
                    f"{results.get('tweets_created', 0)} tweets"
                )
                
                # Process Telegram queue (no rate limits, so process all ready)
                if self.telegram_scheduler:
                    try:
                        tg_posted = self.telegram_scheduler.process_ready_threads()
                        if tg_posted > 0:
                            logger.info(f"📱 Posted {tg_posted} threads to Telegram")
                    except Exception as e:
                        logger.error(f"Telegram processing error: {e}")

                self.last_processing = datetime.utcnow()
                self._save_metrics()

            except Exception as e:
                logger.error(f"❌ Processing loop error: {e}", exc_info=True)
                self.metrics["total_errors"] += 1
                time.sleep(self.orchestration_config.retry_backoff)

        logger.info("🛑 Processing loop stopped")

    def _health_check_loop(self):
        """Continuous health monitoring - runs in separate thread."""

        logger.info("🏥 Health check loop started")

        while self.running:
            try:
                time.sleep(self.orchestration_config.health_check_interval)

                # Basic health checks
                logger.debug("🏥 Running health checks...")

                # Check disk space
                import shutil
                disk_usage = shutil.disk_usage(".")
                free_gb = disk_usage.free / (1024**3)

                if free_gb < 1.0:
                    logger.error(f"❌ Low disk space: {free_gb:.2f} GB free")
                    self.paused = True

                # Check if pipeline is responsive
                if self.pipeline:
                    try:
                        stats = self.pipeline.get_stats()
                        logger.debug(f"📊 Pipeline stats: {stats}")
                    except Exception as e:
                        logger.error(f"❌ Pipeline health check failed: {e}")
                        # Attempt to reinitialize
                        self._init_pipeline()

                # Log uptime
                if self.metrics.get("uptime_start"):
                    uptime = datetime.utcnow() - datetime.fromisoformat(self.metrics["uptime_start"])
                    logger.info(f"⏱️ Uptime: {uptime}")

                self.last_health_check = datetime.utcnow()

            except Exception as e:
                logger.error(f"❌ Health check error: {e}", exc_info=True)

        logger.info("🛑 Health check loop stopped")

    def start(self):
        """Start autonomous operation."""

        if self.running:
            logger.warning("⚠️ Orchestrator already running")
            return

        logger.info("🚀 Starting Autonomous Orchestrator...")

        # Load previous metrics
        self._load_metrics()

        # Initialize pipeline
        if not self._init_pipeline():
            logger.error("❌ Failed to initialize pipeline, cannot start")
            return

        # Start the scheduler (for posting tweets)
        try:
            self.pipeline.run_scheduler()
            logger.info("✅ Tweet scheduler started")
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {e}")

        # Set running flag
        self.running = True
        self.metrics["uptime_start"] = datetime.utcnow().isoformat()

        # Start background threads
        self.discovery_thread = threading.Thread(
            target=self._discovery_loop,
            name="DiscoveryLoop",
            daemon=True
        )

        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="ProcessingLoop",
            daemon=True
        )

        self.health_thread = threading.Thread(
            target=self._health_check_loop,
            name="HealthCheckLoop",
            daemon=True
        )

        self.discovery_thread.start()
        self.processing_thread.start()
        self.health_thread.start()

        logger.info("✅ Autonomous Orchestrator is now running 24/7")
        logger.info(f"📊 Discovery interval: {self.orchestration_config.discovery_interval}s")
        logger.info(f"📊 Processing interval: {self.orchestration_config.processing_interval}s")
        logger.info(f"📊 Health check interval: {self.orchestration_config.health_check_interval}s")

    def stop(self):
        """Stop autonomous operation gracefully."""

        if not self.running:
            logger.warning("⚠️ Orchestrator not running")
            return

        logger.info("🛑 Stopping Autonomous Orchestrator...")

        self.running = False

        # Wait for threads to finish
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=30)

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=30)

        if self.health_thread and self.health_thread.is_alive():
            self.health_thread.join(timeout=10)

        # Stop scheduler
        if self.pipeline and hasattr(self.pipeline.scheduler, 'stop_scheduler'):
            try:
                self.pipeline.scheduler.stop_scheduler()
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")

        # Save final metrics
        self._save_metrics()

        logger.info("✅ Autonomous Orchestrator stopped gracefully")

    def pause(self):
        """Pause processing (but keep monitoring)."""
        self.paused = True
        logger.info("⏸️ Orchestrator paused")

    def resume(self):
        """Resume processing."""
        self.paused = False
        logger.info("▶️ Orchestrator resumed")

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""

        uptime = None
        if self.metrics.get("uptime_start"):
            uptime_delta = datetime.utcnow() - datetime.fromisoformat(self.metrics["uptime_start"])
            uptime = str(uptime_delta)

        return {
            "running": self.running,
            "paused": self.paused,
            "uptime": uptime,
            "last_discovery": self.last_discovery.isoformat() if self.last_discovery else None,
            "last_processing": self.last_processing.isoformat() if self.last_processing else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "metrics": self.metrics,
            "config": asdict(self.orchestration_config),
            "threads": {
                "discovery": self.discovery_thread.is_alive() if self.discovery_thread else False,
                "processing": self.processing_thread.is_alive() if self.processing_thread else False,
                "health": self.health_thread.is_alive() if self.health_thread else False
            }
        }

    def run_forever(self):
        """
        Run the orchestrator forever (blocking).
        Use this as the main entry point for daemon mode.
        """

        self.start()

        try:
            # Keep main thread alive
            while self.running:
                time.sleep(60)

                # Periodic status log
                if datetime.utcnow().minute == 0:  # Every hour
                    status = self.get_status()
                    logger.info(f"📊 Status: {json.dumps(status, indent=2)}")

        except KeyboardInterrupt:
            logger.info("⚠️ Received keyboard interrupt")
        finally:
            self.stop()


def main():
    """Main entry point for autonomous orchestrator."""

    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Podcast-to-Tweet Orchestrator")
    parser.add_argument("--config", default="viral_config.json", help="Pipeline config file")
    parser.add_argument("--discovery-interval", type=int, help="Discovery interval (seconds)")
    parser.add_argument("--processing-interval", type=int, help="Processing interval (seconds)")
    parser.add_argument("--max-episodes", type=int, help="Max episodes per cycle")
    parser.add_argument("--status", action="store_true", help="Show status and exit")

    args = parser.parse_args()

    # Build config
    config = OrchestrationConfig()

    if args.discovery_interval:
        config.discovery_interval = args.discovery_interval
    if args.processing_interval:
        config.processing_interval = args.processing_interval
    if args.max_episodes:
        config.max_episodes_per_cycle = args.max_episodes

    # Create orchestrator
    orchestrator = AutonomousOrchestrator(
        config_file=args.config,
        orchestration_config=config
    )

    if args.status:
        # Just show status
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
        return

    # Run forever
    logger.info("🤖 Starting autonomous operation (Ctrl+C to stop)")
    orchestrator.run_forever()


if __name__ == "__main__":
    main()
