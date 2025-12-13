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

    main()
