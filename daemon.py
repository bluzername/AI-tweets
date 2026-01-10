#!/usr/bin/env python3
"""
Autonomous Daemon - Entry point for 24/7 Docker deployment.

This script runs the podcast-to-tweets pipeline continuously:
- Discovers new episodes every DISCOVERY_INTERVAL seconds (default: 4 hours)
- Processes pending episodes every PROCESSING_INTERVAL seconds (default: 30 min)
- Posts scheduled threads every SCHEDULER_CHECK_INTERVAL seconds (default: 5 min)

Usage:
    python daemon.py                    # Run with defaults
    python daemon.py --status          # Show current status and exit

Environment Variables:
    DISCOVERY_INTERVAL      - Seconds between RSS feed checks (default: 14400 = 4 hours)
    PROCESSING_INTERVAL     - Seconds between processing cycles (default: 1800 = 30 min)
    SCHEDULER_CHECK_INTERVAL - Seconds between posting checks (default: 300 = 5 min)
    MAX_EPISODES_PER_CYCLE  - Max episodes to process per cycle (default: 5)
    MAX_TWEETS_PER_DAY      - Daily tweet limit safety (default: 100)

Required API Keys (in .env):
    OPENROUTER_API_KEY or OPENAI_API_KEY
    USE_OPENROUTER=true (if using OpenRouter)
    MAIN_API_KEY, MAIN_API_SECRET, MAIN_ACCESS_TOKEN, MAIN_ACCESS_TOKEN_SECRET
"""

import os
import sys
import threading
import logging

logger = logging.getLogger(__name__)

# Set default intervals if not specified
os.environ.setdefault("DISCOVERY_INTERVAL", "14400")      # 4 hours
os.environ.setdefault("PROCESSING_INTERVAL", "1800")       # 30 minutes
os.environ.setdefault("SCHEDULER_CHECK_INTERVAL", "300")   # 5 minutes
os.environ.setdefault("MAX_EPISODES_PER_CYCLE", "5")
os.environ.setdefault("WHISPER_DEVICE", "auto")            # auto, cpu, cuda, mps

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Import and run orchestrator
from src.autonomous_orchestrator import main


def run_hebrew_pipeline():
    """Run Hebrew pipeline in background thread."""
    import time
    from datetime import datetime
    try:
        from hebrew_orchestrator import HebrewOrchestrator
        orchestrator = HebrewOrchestrator(config_file="/app/hebrew_config.json")
        logger.info("Hebrew pipeline started")

        # Run without signal handlers (can't use signals in thread)
        orchestrator.running = True
        interval_minutes = 60

        while orchestrator.running:
            try:
                cycle_start = datetime.now()
                stats = orchestrator.run_cycle()
                logger.info(
                    f"Hebrew cycle: {stats['discovered']} discovered, "
                    f"{stats['processed']} processed, {stats['posted']} posted"
                )
                elapsed = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, interval_minutes * 60 - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Hebrew cycle error: {e}")
                time.sleep(60)

    except Exception as e:
        logger.error(f"Hebrew pipeline crashed: {e}")


def run_web_dashboard():
    """Run web dashboard on port 5000."""
    try:
        from src.web_dashboard import app as dashboard_app
        logger.info("Web dashboard starting on port 5000")
        dashboard_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Web dashboard failed: {e}")


if __name__ == "__main__":
    print("""
    ================================================
       PODCASTS TLDR - Autonomous Daemon
    ================================================

    This daemon will run continuously and:
    - Check for new podcast episodes every 4 hours
    - Process pending episodes every 30 minutes
    - Post scheduled threads every 5 minutes

    Press Ctrl+C to stop gracefully.
    ================================================
    """)

    # Start Hebrew pipeline in background thread
    hebrew_thread = threading.Thread(target=run_hebrew_pipeline, daemon=True, name="HebrewPipeline")
    hebrew_thread.start()
    print("    Hebrew pipeline started in background thread")

    # Start web dashboard in background thread
    dashboard_thread = threading.Thread(target=run_web_dashboard, daemon=True, name="WebDashboard")
    dashboard_thread.start()
    print("    Web dashboard running at http://localhost:5000")
    print()

    # Run main English pipeline (blocking)
    main()
