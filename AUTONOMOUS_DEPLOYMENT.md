# Autonomous Deployment Guide

Complete guide for deploying Podcasts TLDR as a fully autonomous 24/7 service.

## 🎯 Overview

The autonomous deployment transforms this pipeline from a manually-triggered script into a **set-and-forget, 24/7 automated content machine** that:

- ✅ **Continuously monitors** RSS feeds for new episodes
- ✅ **Automatically processes** episodes (transcription → analysis → tweet generation)
- ✅ **Intelligently schedules** and posts tweets at optimal times
- ✅ **Self-heals** from failures with automatic recovery
- ✅ **Monitors health** and sends alerts when issues arise
- ✅ **Respects limits** (API costs, rate limits, operating hours)
- ✅ **Runs indefinitely** without human intervention

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Fastest way to get started**

```bash
# 1. Clone and configure
git clone <repo-url>
cd AI-tweets
cp .env.example .env
nano .env  # Add your API keys

# 2. Start the service
docker-compose up -d

# 3. Check logs
docker-compose logs -f podcasts-tldr

# 4. Check status
docker-compose exec podcasts-tldr python src/health_monitor.py
```

### Option 2: Systemd Service (Linux)

**Best for dedicated servers**

```bash
# 1. Install as system service
sudo ./deployment/systemd/install.sh

# 2. Configure API keys
sudo nano /opt/podcasts-tldr/.env

# 3. Start the service
sudo systemctl start podcasts-tldr
sudo systemctl enable podcasts-tldr  # Auto-start on boot

# 4. Monitor
sudo journalctl -u podcasts-tldr -f
```

### Option 3: Manual Python Process

**For development and testing**

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
nano .env  # Add your API keys

# 3. Run
python src/autonomous_orchestrator.py
```

---

## ⚙️ Configuration

### Essential Settings

Edit `.env` with your API credentials:

```bash
# OpenAI (Required)
OPENAI_API_KEY=sk-...

# Twitter/X.com accounts (Required for posting)
MAIN_API_KEY=...
MAIN_API_SECRET=...
MAIN_ACCESS_TOKEN=...
MAIN_ACCESS_TOKEN_SECRET=...
```

### Autonomous Operation Settings

```bash
# How often to check for new episodes (seconds)
DISCOVERY_INTERVAL=3600  # 1 hour

# How often to process pending episodes (seconds)
PROCESSING_INTERVAL=1800  # 30 minutes

# Health checks (seconds)
HEALTH_CHECK_INTERVAL=300  # 5 minutes

# Episodes to process per cycle
MAX_EPISODES_PER_CYCLE=5
```

### Safety Limits

```bash
# Maximum spend per day (USD)
MAX_API_COST_PER_DAY=50.0

# Maximum tweets per day
MAX_TWEETS_PER_DAY=100
```

### Operating Hours (Optional)

Limit when the system operates:

```bash
# Only operate 6 AM - 10 PM UTC (empty = 24/7)
OPERATING_HOURS_START=6
OPERATING_HOURS_END=22

# Pause processing 1 AM - 6 AM UTC (maintenance window)
QUIET_HOURS_START=1
QUIET_HOURS_END=6
```

### Alerting (Optional but Recommended)

Get notified when issues occur:

```bash
# Email alerts
ALERT_EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_TO=admin@example.com

# Slack alerts
ALERT_SLACK_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

---

## 📊 Monitoring & Management

### Health Checks

```bash
# Quick health check
python src/health_monitor.py

# Continuous monitoring
python src/health_monitor.py --watch --interval 60

# Save health report
python src/health_monitor.py --save
```

### View Status

```bash
# Docker
docker-compose exec podcasts-tldr python src/autonomous_orchestrator.py --status

# Systemd
sudo systemctl status podcasts-tldr

# Check metrics
cat data/orchestrator_metrics.json | jq
```

### View Logs

```bash
# Docker
docker-compose logs -f podcasts-tldr

# Systemd
sudo journalctl -u podcasts-tldr -f

# Log files
tail -f logs/autonomous_*.log
```

### Control the Service

```bash
# Docker
docker-compose stop     # Stop
docker-compose start    # Start
docker-compose restart  # Restart

# Systemd
sudo systemctl stop podcasts-tldr
sudo systemctl start podcasts-tldr
sudo systemctl restart podcasts-tldr
```

---

## 🔧 Advanced Configuration

### Custom Check Intervals

Balance between freshness and resource usage:

```bash
# Aggressive (high resource usage)
DISCOVERY_INTERVAL=1800   # Check every 30 min
PROCESSING_INTERVAL=900   # Process every 15 min

# Moderate (recommended)
DISCOVERY_INTERVAL=3600   # Check every hour
PROCESSING_INTERVAL=1800  # Process every 30 min

# Conservative (low resource usage)
DISCOVERY_INTERVAL=7200   # Check every 2 hours
PROCESSING_INTERVAL=3600  # Process every hour
```

### Resource Limits

Prevent runaway resource usage:

```bash
# Memory limit (MB)
MAX_MEMORY_MB=2048  # 2 GB

# CPU limit (%)
MAX_CPU_PERCENT=80  # 80% max
```

### Retry Configuration

Control automatic retry behavior:

```bash
# Max retry attempts
MAX_RETRIES=3

# Initial retry delay (seconds, doubles each attempt)
RETRY_DELAY=60  # 60s, then 120s, then 240s
```

---

## 🛡️ Self-Healing Features

The system automatically recovers from:

### Database Issues
- **Detection**: Integrity checks on startup and periodically
- **Recovery**: Automatic backup restoration or repair
- **Fallback**: Recreate database if repair fails

### API Failures
- **Retry**: Exponential backoff (3 attempts by default)
- **Circuit breaker**: Pause after repeated failures
- **Alerts**: Notify when critical APIs are down

### Transcription Failures
- **Fallback chain**: YouTube → Local Whisper → Whisper API
- **Caching**: Failed attempts cached to avoid retries
- **Skipping**: Episodes that fail all methods are skipped

### Resource Exhaustion
- **Memory**: Pause processing if memory exceeds threshold
- **Disk**: Stop if disk space < 1 GB
- **Cost**: Stop if daily API cost exceeded

### System Crashes
- **Auto-restart**: Systemd/Docker automatically restarts process
- **State recovery**: Resume from last known good state
- **Transaction rollback**: Database changes rolled back on failure

---

## 🔔 Alerting

### Alert Levels

- **INFO**: Normal operations, milestone events
- **WARNING**: Issues that don't stop operation (slow API, high memory)
- **CRITICAL**: System failures requiring immediate attention

### Alert Channels

Configure one or more:

1. **Email**: Via SMTP (Gmail, SendGrid, etc.)
2. **Slack**: Via webhook
3. **Custom Webhook**: POST JSON to your endpoint
4. **Logs**: Always logged

### Alert Throttling

Prevents alert spam:
- Same alert won't trigger again within `ALERT_THROTTLE_MINUTES` (default: 30)
- Protects against notification storms

### Testing Alerts

```bash
# Test email
python src/alerting.py --test-email

# Test Slack
python src/alerting.py --test-slack

# Test all channels
python src/alerting.py --test-all

# View alert history
python src/alerting.py --stats
```

---

## 📈 Metrics & Analytics

### Orchestrator Metrics

Stored in `data/orchestrator_metrics.json`:

```json
{
  "total_discoveries": 150,
  "total_processed": 142,
  "total_tweets_created": 568,
  "total_errors": 8,
  "api_cost_today": 12.45,
  "tweets_today": 24,
  "uptime_start": "2025-01-15T08:00:00",
  "cycles_completed": 450
}
```

### Health Reports

Stored in `data/health_report.json`:

```json
{
  "overall_status": "healthy",
  "checks": [
    {"name": "disk_space", "status": "healthy", "message": "..."},
    {"name": "memory", "status": "healthy", "message": "..."},
    {"name": "openai_api", "status": "healthy", "message": "..."}
  ],
  "summary": {
    "healthy": 7,
    "warnings": 0,
    "critical": 0
  }
}
```

### Recovery History

Stored in `data/recovery_history.json`:

Tracks all automatic recovery actions taken.

---

## 🐳 Docker Deployment Details

### Build & Run

```bash
# Build image
docker-compose build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Persistent Data

Data is stored in volumes:

```
./data      → Episode database, metrics
./logs      → Application logs
./output    → Generated markdown files
./cache     → Transcription cache
./backups   → Database backups
```

### Resource Limits

Configured in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
```

### Updates

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

---

## 🖥️ Systemd Deployment Details

### Installation

The `deployment/systemd/install.sh` script:

1. Creates `/opt/podcasts-tldr` directory
2. Creates `podcasts` service user
3. Copies application files
4. Sets up Python virtual environment
5. Installs systemd service

### Service Management

```bash
# Status
sudo systemctl status podcasts-tldr

# Start/Stop
sudo systemctl start podcasts-tldr
sudo systemctl stop podcasts-tldr

# Enable auto-start on boot
sudo systemctl enable podcasts-tldr

# Disable auto-start
sudo systemctl disable podcasts-tldr

# View logs
sudo journalctl -u podcasts-tldr -f
sudo journalctl -u podcasts-tldr --since "1 hour ago"
```

### File Locations

- **Application**: `/opt/podcasts-tldr/`
- **Config**: `/opt/podcasts-tldr/.env`
- **Data**: `/opt/podcasts-tldr/data/`
- **Logs**: `/opt/podcasts-tldr/logs/`
- **Service File**: `/etc/systemd/system/podcasts-tldr.service`

### Updating

```bash
# Stop service
sudo systemctl stop podcasts-tldr

# Update code
cd /opt/podcasts-tldr
sudo -u podcasts git pull

# Install dependencies
sudo -u podcasts venv/bin/pip install -r requirements.txt

# Restart
sudo systemctl start podcasts-tldr
```

---

## 🧪 Testing

### Before Production

1. **Test basic pipeline**:
   ```bash
   python viral_main.py --mode full --episode-limit 1
   ```

2. **Test autonomous orchestrator**:
   ```bash
   # Run for 5 minutes then Ctrl+C
   python src/autonomous_orchestrator.py --discovery-interval 60 --processing-interval 120
   ```

3. **Test health monitoring**:
   ```bash
   python src/health_monitor.py
   ```

4. **Test recovery**:
   ```bash
   python src/auto_recovery.py --check-databases
   ```

5. **Test alerting**:
   ```bash
   python src/alerting.py --test-all
   ```

### Dry-Run Mode

To test without posting to Twitter:

```bash
# In .env
DRY_RUN=true
```

This will:
- ✅ Discover episodes
- ✅ Transcribe audio
- ✅ Generate tweets
- ✅ Save to markdown files
- ❌ NOT post to Twitter

---

## 🔒 Security Best Practices

### API Keys

- ✅ Use `.env` file (never commit to git)
- ✅ Restrict file permissions: `chmod 600 .env`
- ✅ Use separate keys for testing vs production
- ✅ Rotate keys periodically

### System Access

- ✅ Run as non-root user (`podcasts` user in systemd)
- ✅ Limit file system access (Docker/systemd sandboxing)
- ✅ Use read-only mounts where possible

### Monitoring

- ✅ Enable alerting for critical events
- ✅ Monitor API costs daily
- ✅ Review logs for suspicious activity
- ✅ Set up log rotation

---

## 🐛 Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs podcasts-tldr
# OR
sudo journalctl -u podcasts-tldr -n 50

# Common issues:
# - Missing API keys in .env
# - Invalid config in viral_config.json
# - Permission issues (systemd)
```

### High Memory Usage

```bash
# Check current usage
docker stats podcasts-tldr
# OR
ps aux | grep autonomous_orchestrator

# Solutions:
# - Reduce MAX_EPISODES_PER_CYCLE
# - Increase PROCESSING_INTERVAL
# - Lower MAX_MEMORY_MB threshold
```

### API Errors

```bash
# Check OpenAI API status
python -c "from openai import OpenAI; OpenAI(api_key='your-key').models.list()"

# Check rate limits in logs
grep -i "rate limit" logs/*.log

# Solutions:
# - Verify API key is valid
# - Check OpenAI account billing
# - Reduce processing frequency
```

### Database Locked

```bash
# Check for multiple instances
ps aux | grep autonomous_orchestrator

# Kill duplicates
killall -9 autonomous_orchestrator  # Only if safe!

# Restart service
docker-compose restart
# OR
sudo systemctl restart podcasts-tldr
```

### No Episodes Being Processed

```bash
# Check RSS feeds are accessible
python -c "import feedparser; print(feedparser.parse('https://...'))"

# Check database
sqlite3 data/episodes.db "SELECT COUNT(*) FROM episodes WHERE processing_status='pending';"

# Check orchestrator metrics
cat data/orchestrator_metrics.json | jq
```

---

## 📞 Support & Maintenance

### Regular Maintenance

**Daily**:
- Check metrics dashboard
- Review alert history

**Weekly**:
- Review logs for errors
- Check disk space usage
- Verify tweet quality

**Monthly**:
- Rotate API keys
- Clean old logs and backups
- Review and optimize config
- Update dependencies

### Backup Strategy

**Automatic backups** (handled by auto_recovery.py):
- Database snapshots before operations
- Kept for 7 days
- Location: `backups/`

**Manual backup**:
```bash
# Backup everything
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/ output/

# Backup just databases
python src/auto_recovery.py --backup-all
```

### Updates

```bash
# Docker
git pull
docker-compose down
docker-compose build
docker-compose up -d

# Systemd
sudo systemctl stop podcasts-tldr
cd /opt/podcasts-tldr
sudo -u podcasts git pull
sudo -u podcasts venv/bin/pip install -r requirements.txt
sudo systemctl start podcasts-tldr
```

---

## 🎓 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Autonomous Orchestrator                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Discovery   │  │  Processing  │  │    Health    │     │
│  │     Loop     │  │     Loop     │  │  Monitoring  │     │
│  │   (1 hour)   │  │  (30 min)    │  │   (5 min)    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐
│  RSS Feeds       │  │  Pipeline        │  │  Alerts      │
│  • Check new     │  │  • Transcribe    │  │  • Email     │
│  • Viral score   │  │  • Analyze       │  │  • Slack     │
│  • Store DB      │  │  • Generate      │  │  • Webhook   │
└──────────────────┘  │  • Schedule      │  └──────────────┘
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │  Auto-Recovery   │
                      │  • DB repair     │
                      │  • Retry logic   │
                      │  • Rollback      │
                      └──────────────────┘
```

---

## 🎯 Next Steps After Deployment

Once your autonomous system is running:

1. **Monitor for 24 hours** to ensure stability
2. **Review generated content** quality
3. **Adjust intervals** based on volume
4. **Enable alerting** for peace of mind
5. **Set up dashboard** (coming in Sprint 4)
6. **Expand to more podcasts** gradually
7. **Consider multi-platform** (coming in Sprint 5)

---

## 🚧 Coming Soon

Future autonomous features (Sprints 2-6):

- **Sprint 2**: Multi-model AI ensemble, better insight quality
- **Sprint 3**: Performance tracking & A/B testing automation
- **Sprint 4**: Web dashboard for monitoring
- **Sprint 5**: Multi-platform publishing (LinkedIn, Threads)
- **Sprint 6**: ML-powered viral prediction

---

## 📝 License

See main repository LICENSE file.
