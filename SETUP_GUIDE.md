# Complete Setup and Testing Guide
## Podcast Highlights to Twitter Threads - Autonomous System

This guide will walk you through setting up and testing the autonomous podcast-to-Twitter system from scratch. No prior experience assumed.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [API Keys Configuration](#api-keys-configuration)
4. [Installation](#installation)
5. [Database Initialization](#database-initialization)
6. [Configuration](#configuration)
7. [Testing the System](#testing-the-system)
8. [Running the Autonomous System](#running-the-autonomous-system)
9. [Monitoring and Dashboard](#monitoring-and-dashboard)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Install Python 3.11+

**Windows:**
1. Download Python from https://www.python.org/downloads/
2. Run the installer
3. **IMPORTANT:** Check "Add Python to PATH" during installation
4. Verify installation:
   ```bash
   python --version
   ```
   Should show: `Python 3.11.x` or higher

**macOS:**
```bash
# Using Homebrew
brew install python@3.11

# Verify
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Verify
python3 --version
```

### 2. Install Git

**Windows:** Download from https://git-scm.com/download/win

**macOS:**
```bash
brew install git
```

**Linux:**
```bash
sudo apt install git
```

Verify:
```bash
git --version
```

### 3. Get API Keys

You'll need accounts and API keys for:

1. **OpenAI** (required for transcription and AI analysis)
   - Sign up at https://platform.openai.com/
   - Go to API Keys section
   - Create new secret key
   - Copy and save it (you won't see it again!)

2. **Anthropic Claude** (optional, for multi-model AI)
   - Sign up at https://console.anthropic.com/
   - Navigate to API Keys
   - Create new key
   - Save the key

3. **Google AI** (optional, for multi-model AI)
   - Go to https://makersuite.google.com/app/apikey
   - Create API key
   - Save the key

4. **Twitter/X API** (required for posting)
   - Go to https://developer.twitter.com/
   - Create a new app
   - Generate API keys (need all 4):
     - API Key
     - API Secret
     - Access Token
     - Access Token Secret

---

## Initial Setup

### Step 1: Clone the Repository

Open your terminal/command prompt and run:

```bash
# Navigate to where you want the project
cd ~/Documents  # or C:\Users\YourName\Documents on Windows

# Clone the repository
git clone https://github.com/yourusername/AI-tweets.git

# Enter the directory
cd AI-tweets
```

### Step 2: Create Virtual Environment

This keeps the project dependencies isolated:

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your command prompt.

---

## API Keys Configuration

### Step 1: Create .env File

```bash
# Copy the example file
cp .env.example .env

# On Windows, use:
copy .env.example .env
```

### Step 2: Edit .env File

Open `.env` in your favorite text editor (Notepad, VSCode, nano, vim, etc.)

```bash
# macOS/Linux
nano .env

# Windows
notepad .env

# Or use any code editor
code .env  # VSCode
```

### Step 3: Fill in Your API Keys

Replace the placeholder values with your actual API keys:

```bash
# === REQUIRED API KEYS ===

# OpenAI (REQUIRED)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Twitter API (REQUIRED for posting)
TWITTER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxx
TWITTER_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWITTER_ACCESS_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWITTER_ACCESS_TOKEN_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# === OPTIONAL API KEYS (for enhanced features) ===

# Anthropic Claude (for multi-model AI)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google AI (for multi-model AI)
GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# === AUTONOMOUS OPERATION SETTINGS ===

# How often to check for new episodes (in seconds)
RSS_CHECK_INTERVAL=3600  # 1 hour

# How often to process pending episodes (in seconds)
PROCESSING_INTERVAL=1800  # 30 minutes

# How often to check system health (in seconds)
HEALTH_CHECK_INTERVAL=300  # 5 minutes

# Maximum concurrent processing tasks
MAX_CONCURRENT_TASKS=3

# Operating hours (24-hour format, leave empty for 24/7)
OPERATING_HOURS_START=  # e.g., 08 for 8 AM
OPERATING_HOURS_END=    # e.g., 22 for 10 PM

# Quiet hours (no posting during these times)
QUIET_HOURS_START=  # e.g., 23 for 11 PM
QUIET_HOURS_END=    # e.g., 07 for 7 AM

# === ALERTING CONFIGURATION ===

# Email alerts (optional)
ALERT_EMAIL_ENABLED=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_TO=admin@yourdomain.com

# Slack alerts (optional)
ALERT_SLACK_ENABLED=false
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# === COST & BUDGET CONTROLS ===

# Daily budget limit (USD)
DAILY_BUDGET_LIMIT=10.00

# Monthly budget limit (USD)
MONTHLY_BUDGET_LIMIT=200.00

# Stop processing if budget exceeded
STOP_ON_BUDGET_EXCEEDED=true

# === RATE LIMITING ===

# These are set to safe defaults, adjust only if needed
OPENAI_REQUESTS_PER_MINUTE=60
ANTHROPIC_REQUESTS_PER_MINUTE=50
GOOGLE_REQUESTS_PER_MINUTE=60
TWITTER_REQUESTS_PER_MINUTE=50
```

**Save the file** (Ctrl+O then Ctrl+X in nano, or Ctrl+S in most editors)

---

## Installation

### Step 1: Install Python Dependencies

Make sure your virtual environment is activated (you should see `(venv)`):

```bash
pip install -r requirements.txt
```

This will take 2-5 minutes. You should see packages being downloaded and installed.

**If you get errors:**
- On Linux, you might need: `sudo apt install python3-dev build-essential`
- On macOS, you might need: `xcode-select --install`

### Step 2: Verify Installation

```bash
# Check that key packages are installed
python -c "import openai; print('OpenAI:', openai.__version__)"
python -c "import anthropic; print('Anthropic:', anthropic.__version__)"
python -c "import tweepy; print('Tweepy:', tweepy.__version__)"
```

All should print version numbers without errors.

---

## Database Initialization

The system uses SQLite databases. They'll be created automatically, but let's verify the structure:

### Step 1: Check Data Directory

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Check it exists
ls -la data/
```

### Step 2: Test Database Creation

Run this Python script to initialize databases:

```bash
python -c "
import sqlite3
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Initialize each database
databases = [
    'data/episodes.db',
    'data/insights.db',
    'data/tweet_queue.db',
    'data/performance_metrics.db',
    'data/costs.db',
    'data/ab_tests.db'
]

for db_path in databases:
    conn = sqlite3.connect(db_path)
    conn.close()
    print(f'✓ Created: {db_path}')
"
```

You should see checkmarks for each database.

---

## Configuration

### Step 1: Create Podcast Configuration

Create or edit `viral_config.json`:

```bash
# Copy example if it exists
cp config.example.json viral_config.json 2>/dev/null || true

# Or create new one
nano viral_config.json
```

**Minimal configuration:**

```json
{
  "podcasts": [
    {
      "name": "Lex Fridman Podcast",
      "rss_url": "https://lexfridman.com/feed/podcast/",
      "enabled": true,
      "priority": "high"
    },
    {
      "name": "Huberman Lab",
      "rss_url": "https://feeds.megaphone.fm/hubermanlab",
      "enabled": true,
      "priority": "high"
    },
    {
      "name": "All-In Podcast",
      "rss_url": "https://feeds.megaphone.fm/all-in-podcast",
      "enabled": true,
      "priority": "medium"
    }
  ],
  "twitter_accounts": [
    {
      "name": "main",
      "enabled": true,
      "audience": "tech professionals and entrepreneurs",
      "tone": "authoritative yet accessible",
      "use_emojis": true,
      "max_hashtags": 3
    }
  ],
  "ai_models": {
    "transcription": "whisper-1",
    "analysis": {
      "enabled_models": ["gpt-4-turbo-preview", "claude-3-sonnet", "gemini-pro"],
      "consensus_threshold": 0.7
    }
  },
  "content_settings": {
    "max_insights_per_episode": 3,
    "min_insight_length": 100,
    "hook_variants_per_insight": 5,
    "enable_deduplication": true,
    "similarity_threshold": 0.85
  }
}
```

Save the file.

---

## Testing the System

Now let's test each component step by step.

### Test 1: Health Monitor

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from health_monitor import HealthMonitor

monitor = HealthMonitor()
status, checks = monitor.check_all()

print(f'\n=== HEALTH CHECK RESULTS ===')
print(f'Overall Status: {status.value.upper()}\n')

for check in checks:
    icon = '✓' if check.status.value == 'healthy' else '⚠' if check.status.value == 'warning' else '✗'
    print(f'{icon} {check.component}: {check.status.value}')
    print(f'  Message: {check.message}')
    if check.details:
        print(f'  Details: {check.details}')
    print()
"
```

**Expected output:** All checks should show as healthy or warning (not critical).

### Test 2: Rate Limiter

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from rate_limiter import RateLimiter

limiter = RateLimiter()

print('\n=== RATE LIMITER TEST ===')
print('Testing OpenAI rate limits...')

# Simulate 3 API calls
for i in range(3):
    wait_time = limiter.wait_if_needed('openai')
    print(f'Call {i+1}: waited {wait_time:.2f}s')
    limiter.record_request('openai')

status = limiter.get_status('openai')
print(f'\nCurrent status:')
print(f'  Requests in last minute: {status[\"requests_last_minute\"]}')
print(f'  Available: {status[\"available_requests_minute\"]}')
print('✓ Rate limiter working correctly')
"
```

**Expected output:** Should show 3 calls recorded, no errors.

### Test 3: Cost Tracker

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from cost_tracker import CostTracker

tracker = CostTracker()

print('\n=== COST TRACKER TEST ===')

# Track some sample costs
tracker.track_cost(
    service='openai',
    operation='transcription',
    input_tokens=0,
    output_tokens=0,
    metadata={'audio_minutes': 60}
)

tracker.track_cost(
    service='openai',
    operation='analysis',
    input_tokens=5000,
    output_tokens=1000,
    metadata={'model': 'gpt-4-turbo-preview'}
)

print('Tracked sample costs:')
breakdown = tracker.get_cost_breakdown()
print(f'  Total cost: \${breakdown[\"total_cost\"]:.4f}')
print(f'  Number of operations: {breakdown[\"total_operations\"]}')

# Check budget
within_budget, details = tracker.check_budget()
print(f'\nBudget status: {\"✓ Within budget\" if within_budget else \"✗ Over budget\"}')
print(f'  Daily spent: \${details[\"daily_spent\"]:.2f} / \${details[\"daily_limit\"]:.2f}')
print(f'  Monthly spent: \${details[\"monthly_spent\"]:.2f} / \${details[\"monthly_limit\"]:.2f}')
"
```

**Expected output:** Should show costs tracked and budget status.

### Test 4: Multi-Model AI Analyzer

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from multi_model_analyzer import MultiModelAnalyzer

# This will use only models with valid API keys
analyzer = MultiModelAnalyzer()

print('\n=== MULTI-MODEL ANALYZER TEST ===')
print(f'Enabled models: {analyzer.enabled_models}')
print(f'Model count: {len(analyzer.enabled_models)}')

if len(analyzer.enabled_models) == 0:
    print('⚠ No AI models enabled. Check your API keys in .env')
else:
    print('✓ AI models initialized')
"
```

**Expected output:** Should show which AI models are enabled based on your API keys.

### Test 5: Hook Generator

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from hook_generator import HookGenerator

generator = HookGenerator()

print('\n=== HOOK GENERATOR TEST ===')

test_insight = 'Sleep is crucial for memory consolidation. Studies show that 7-9 hours of quality sleep improves learning by 40%.'

print(f'Test insight: {test_insight}\n')

# Generate hooks
hooks = generator.generate_variants(
    insight=test_insight,
    podcast_name='Test Podcast',
    target_count=3
)

print(f'Generated {len(hooks)} hook variants:\n')
for i, hook in enumerate(hooks, 1):
    print(f'{i}. Pattern: {hook.pattern.value}')
    print(f'   Text: {hook.text}')
    print(f'   Score: {hook.score:.2f}')
    print()

print('✓ Hook generator working correctly')
"
```

**Expected output:** Should generate 3 different hook variants.

### Test 6: Performance Tracker

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from performance_tracker import PerformanceTracker, TweetPerformance

tracker = PerformanceTracker()

print('\n=== PERFORMANCE TRACKER TEST ===')

# Save a test performance
perf = TweetPerformance(
    tweet_id='test_123',
    content='Test tweet',
    posted_at='2025-01-19T12:00:00',
    likes=100,
    retweets=50,
    replies=10,
    impressions=5000,
    engagement_rate=3.2,
    hook_pattern='question',
    podcast_name='Test Podcast'
)

tracker.save_performance(perf)
print('✓ Saved test performance data')

# Get statistics
stats = tracker.get_statistics()
print(f'\nStatistics:')
print(f'  Total tweets tracked: {stats[\"total_tweets\"]}')
print(f'  Average engagement: {stats[\"avg_engagement_rate\"]:.2f}%')
print(f'  Average viral score: {stats[\"avg_viral_score\"]:.2f}')
"
```

**Expected output:** Should save and retrieve performance data.

### Test 7: Run Full Test Suite

```bash
# Run all automated tests
pytest tests/ -v --tb=short

# Or run specific test files
pytest tests/test_quality.py -v
pytest tests/test_performance.py -v
```

**Expected output:** Should see tests passing. Some tests may be slow (30+ seconds).

---

## Running the Autonomous System

### Option 1: Manual Test Run

Test the orchestrator without running 24/7:

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from autonomous_orchestrator import AutonomousOrchestrator

print('Starting test run of autonomous orchestrator...')
print('This will check for new episodes and process one cycle.\n')

orchestrator = AutonomousOrchestrator('viral_config.json')

# Run discovery once
print('=== RSS Discovery Cycle ===')
orchestrator._discovery_loop_single_cycle()

# Run processing once
print('\n=== Processing Cycle ===')
orchestrator._processing_loop_single_cycle()

# Run health check
print('\n=== Health Check ===')
orchestrator._health_check_loop_single_cycle()

print('\n✓ Test cycle complete!')
print('Check data/ directory for generated databases and logs.')
"
```

**Expected output:** Should discover episodes from RSS feeds and process them.

### Option 2: Run Full Autonomous Mode

**WARNING:** This runs 24/7 until you stop it (Ctrl+C).

```bash
# Start the autonomous orchestrator
python -c "
import sys
sys.path.insert(0, 'src')
from autonomous_orchestrator import AutonomousOrchestrator

print('Starting autonomous orchestrator in 24/7 mode...')
print('Press Ctrl+C to stop.\n')

orchestrator = AutonomousOrchestrator('viral_config.json')
orchestrator.start()
"
```

To stop: Press `Ctrl+C`

### Option 3: Run as Background Service (Linux/macOS)

```bash
# Start in background
nohup python -c "
import sys
sys.path.insert(0, 'src')
from autonomous_orchestrator import AutonomousOrchestrator
orchestrator = AutonomousOrchestrator('viral_config.json')
orchestrator.start()
" > autonomous.log 2>&1 &

# Check it's running
ps aux | grep autonomous

# View logs
tail -f autonomous.log

# Stop it
pkill -f autonomous_orchestrator
```

### Option 4: Run with Docker

```bash
# Build the image
docker build -t podcasts-tldr .

# Run the container
docker run -d \
  --name podcasts-autonomous \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/viral_config.json:/app/viral_config.json \
  -p 5050:5050 \
  podcasts-tldr

# Check logs
docker logs -f podcasts-autonomous

# Stop
docker stop podcasts-autonomous
```

---

## Monitoring and Dashboard

### Start the Web Dashboard

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from web_dashboard import app

print('Starting web dashboard...')
print('Open http://localhost:5050 in your browser\n')

app.run(host='0.0.0.0', port=5050, debug=False)
"
```

**Or run in background:**

```bash
nohup python -c "
import sys
sys.path.insert(0, 'src')
from web_dashboard import app
app.run(host='0.0.0.0', port=5050, debug=False)
" > dashboard.log 2>&1 &
```

### Access the Dashboard

Open your web browser and go to:
```
http://localhost:5050
```

You should see:
- System health status (green/yellow/red indicators)
- Performance metrics (total tweets, engagement rates)
- A/B testing results
- Optimization recommendations
- Real-time updates every 30 seconds

### API Endpoints

You can also query the system programmatically:

```bash
# Get system status
curl http://localhost:5050/api/status | python -m json.tool

# Get performance metrics
curl http://localhost:5050/api/performance | python -m json.tool

# Get A/B test results
curl http://localhost:5050/api/ab-tests | python -m json.tool

# Get optimization recommendations
curl http://localhost:5050/api/optimizations | python -m json.tool
```

---

## Troubleshooting

### Problem: "No module named 'openai'"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Problem: "API key not found" errors

**Solution:**
1. Check `.env` file exists: `ls -la .env`
2. Verify API keys are set correctly (no quotes needed, no spaces)
3. Make sure `.env` is in the same directory as your scripts

```bash
# Test that .env is being read
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('OPENAI_API_KEY set:', 'OPENAI_API_KEY' in os.environ)
print('First 10 chars:', os.getenv('OPENAI_API_KEY', '')[:10])
"
```

### Problem: "Database is locked"

**Solution:**
```bash
# Stop all running instances
pkill -f autonomous_orchestrator
pkill -f web_dashboard

# Wait a moment
sleep 2

# Try again
```

### Problem: Tests are very slow or hanging

**Solution:**
Some tests make actual API calls or have long timeouts. Run specific test groups:

```bash
# Run only fast tests
pytest tests/test_quality.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Problem: "Permission denied" when creating directories

**Solution:**
```bash
# Create data directory with proper permissions
mkdir -p data
chmod 755 data

# Or run with sudo (not recommended)
sudo mkdir -p data
sudo chown $USER:$USER data
```

### Problem: Docker build fails

**Solution:**
```bash
# Clean up old images
docker system prune -a

# Build with no cache
docker build --no-cache -t podcasts-tldr .

# Check Docker has enough memory (needs at least 2GB)
docker info | grep Memory
```

### Problem: RSS feeds not being discovered

**Solution:**
1. Check `viral_config.json` has valid RSS URLs
2. Test URLs manually:
   ```bash
   curl -I https://lexfridman.com/feed/podcast/
   ```
3. Check internet connection
4. Review logs:
   ```bash
   tail -f data/orchestrator_metrics.json
   ```

### Problem: Tweets not posting

**Solution:**
1. Verify Twitter API credentials:
   ```bash
   python -c "
   import os
   from dotenv import load_dotenv
   load_dotenv()

   keys = ['TWITTER_API_KEY', 'TWITTER_API_SECRET',
           'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET']

   for key in keys:
       value = os.getenv(key, '')
       print(f'{key}: {\"SET\" if value else \"MISSING\"}')
   "
   ```

2. Test Twitter connection:
   ```bash
   python -c "
   import tweepy
   import os
   from dotenv import load_dotenv
   load_dotenv()

   client = tweepy.Client(
       consumer_key=os.getenv('TWITTER_API_KEY'),
       consumer_secret=os.getenv('TWITTER_API_SECRET'),
       access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
       access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
   )

   me = client.get_me()
   print(f'Connected as: {me.data.username}')
   "
   ```

3. Check if fallback .md files are being created in `output/` directory

### Problem: High API costs

**Solution:**
1. Check cost tracker:
   ```bash
   python -c "
   import sys
   sys.path.insert(0, 'src')
   from cost_tracker import CostTracker

   tracker = CostTracker()
   breakdown = tracker.get_cost_breakdown()

   print('Cost breakdown:')
   for service, data in breakdown.get('by_service', {}).items():
       print(f'  {service}: \${data[\"cost\"]:.2f}')
   "
   ```

2. Reduce processing frequency in `.env`:
   ```bash
   RSS_CHECK_INTERVAL=7200  # Check every 2 hours instead of 1
   PROCESSING_INTERVAL=3600  # Process every hour instead of 30 min
   ```

3. Limit number of podcasts in `viral_config.json`

4. Set lower daily budget:
   ```bash
   DAILY_BUDGET_LIMIT=5.00
   ```

### Problem: Dashboard not loading

**Solution:**
1. Check if Flask is running:
   ```bash
   ps aux | grep web_dashboard
   ```

2. Check the port isn't in use:
   ```bash
   lsof -i :5050
   # If something is using it, kill it:
   kill -9 <PID>
   ```

3. Try a different port:
   ```bash
   python -c "
   import sys
   sys.path.insert(0, 'src')
   from web_dashboard import app
   app.run(host='0.0.0.0', port=8080, debug=True)
   "
   ```

4. Check firewall settings allow port 5050

---

## Next Steps

Once everything is working:

1. **Monitor the system** via the dashboard at http://localhost:5050

2. **Review generated content** in the `output/` directory

3. **Check performance metrics** in `data/performance_metrics.db`

4. **Adjust configuration** in `viral_config.json` based on results

5. **Set up alerts** by configuring email/Slack in `.env`

6. **Deploy to production** using Docker or systemd service

7. **Review costs regularly** via the cost tracker

8. **Optimize based on feedback** from the A/B testing framework

---

## Getting Help

- **Check logs:** Look in `autonomous.log`, `dashboard.log`, and `data/orchestrator_metrics.json`
- **Review test output:** Run `pytest tests/ -v` to see detailed test results
- **Check health status:** Run the health monitor test from the Testing section
- **Verify API keys:** Make sure all required keys are set in `.env`
- **Read error messages carefully:** Most errors indicate exactly what's wrong

---

## Summary Checklist

Before running in production, verify:

- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with all required API keys
- [ ] `viral_config.json` configured with your podcasts
- [ ] Data directory exists and is writable
- [ ] All tests pass (`pytest tests/`)
- [ ] Health checks pass
- [ ] Dashboard accessible at http://localhost:5050
- [ ] Test run completes successfully
- [ ] Budget limits set appropriately
- [ ] Alerts configured (optional)

**Once all items are checked, you're ready to run the autonomous system!**

Good luck! 🚀
