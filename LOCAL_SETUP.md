# Local Setup Guide - Free YouTube Transcription

## Quick Local Setup (5 minutes)

### Step 1: Clone to Your Local Machine
```bash
# On your home/office computer (not cloud server):
git clone https://github.com/bluzername/AI-tweets.git
cd AI-tweets
```

### Step 2: Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure for Local Use
```bash
# Copy environment template
cp .env.example .env

# Edit .env file (use nano, vim, or any text editor)
nano .env
```

**Add your OpenAI API key to .env:**
```bash
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here

# Cost Optimization (FREE YouTube transcripts)
TRANSCRIPTION_METHODS=youtube,whisper

# Pipeline Configuration  
DAYS_BACK=7
MAX_HIGHLIGHTS=3
DRY_RUN=true

# Other settings...
WHISPER_MODEL=whisper-1
GPT_MODEL=gpt-4o-mini
```

### Step 4: Test Local YouTube Access
```bash
# Test if YouTube transcripts work locally
python -c "
from youtube_transcript_api import YouTubeTranscriptApi
try:
    result = YouTubeTranscriptApi().fetch('dQw4w9WgXcQ', ['en'])
    print('✅ YouTube transcripts working locally!')
    print(f'Got {len(result.transcript)} transcript segments')
except Exception as e:
    print(f'❌ Still blocked: {e}')
"
```

### Step 5: Generate Your First Thread
```bash
# Generate thread with FREE YouTube transcription
python main.py --dry-run --limit 1

# Check the generated thread
ls output/threads/
cat output/threads/*.md
```

## Detailed Local Configuration

### Optimized .env for Local Development
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_actual_openai_api_key_here
WHISPER_MODEL=whisper-1
GPT_MODEL=gpt-4o-mini

# COST OPTIMIZATION: Try YouTube first (FREE)
TRANSCRIPTION_METHODS=youtube,whisper

# Pipeline Settings
DAYS_BACK=7          # Look back 7 days for episodes
MAX_HIGHLIGHTS=3     # Extract 3 key insights
DRY_RUN=true        # Save to markdown (don't post yet)

# Optional X.com API (for direct posting later)
# MAIN_API_KEY=your_x_api_key
# MAIN_API_SECRET=your_x_api_secret
# MAIN_ACCESS_TOKEN=your_x_access_token
# MAIN_ACCESS_TOKEN_SECRET=your_x_access_token_secret
# MAIN_BEARER_TOKEN=your_x_bearer_token
```

### Local-Optimized config.json
```json
{
  "podcasts": [
    {
      "name": "The Tim Ferriss Show",
      "rss_url": "https://rss.art19.com/tim-ferriss-show",
      "x_handle": "tferriss",
      "creator_handles": ["tferriss"],
      "categories": ["productivity", "entrepreneurship"]
    },
    {
      "name": "Lex Fridman Podcast",
      "rss_url": "https://lexfridman.com/feed/podcast/",
      "x_handle": "lexfridman", 
      "creator_handles": ["lexfridman"],
      "categories": ["AI", "science", "philosophy"]
    }
  ],
  "accounts": [
    {
      "name": "main",
      "target_audience": "Entrepreneurs and tech professionals",
      "tone": "conversational",
      "emoji_usage": "moderate",
      "hashtag_style": "trending"
    }
  ]
}
```

## Local Usage Commands

### Basic Usage
```bash
# Generate threads for all recent episodes (FREE)
python main.py --dry-run --limit 3

# Process specific podcast
python main.py --dry-run --podcast "The Tim Ferriss Show" --limit 1

# Process multiple podcasts
python main.py --dry-run --limit 1 --podcast "Lex Fridman Podcast"
```

### Cost Monitoring
```bash
# YouTube-only mode (completely FREE)
TRANSCRIPTION_METHODS=youtube python main.py --dry-run --limit 5

# Check what you'd save
echo "YouTube transcripts: FREE"
echo "Whisper alternative: 5 episodes × $0.36 = $1.80"
echo "Monthly savings: ~$54 for daily threads"
```

### Advanced Local Commands
```bash
# Generate content for the entire week
python main.py --dry-run --limit 7

# Test different account styles
python main.py --dry-run --limit 1 --config professional_config.json

# Process older episodes
DAYS_BACK=30 python main.py --dry-run --limit 2
```

## Local Development Workflow

### Daily Content Generation
```bash
#!/bin/bash
# daily_local_content.sh

echo "🌅 Generating morning content..."
python main.py --podcast "The Tim Ferriss Show" --dry-run --limit 1

echo "🌆 Generating evening content..."
python main.py --podcast "Lex Fridman Podcast" --dry-run --limit 1

echo "✅ Content generated! Check output/threads/"
```

### Weekly Batch Processing
```bash
#!/bin/bash
# weekly_batch.sh

echo "📡 Processing all podcasts for the week..."
python main.py --dry-run --limit 10

echo "📊 Summary:"
ls output/threads/ | wc -l | echo "$(cat) thread files generated"
echo "💰 Cost: $0.00 (YouTube transcripts)"
echo "⏱️  Time saved: ~5 hours of manual work"
```

## Local vs Cloud Comparison

| Feature | Local Machine | Cloud Server |
|---------|---------------|--------------|
| **YouTube Transcripts** | ✅ FREE | ❌ Blocked |
| **Cost per Episode** | $0.00 | $0.36 |
| **Monthly Cost (30 episodes)** | $0 | $11 |
| **Setup Complexity** | Simple | Simple |
| **Processing Speed** | Instant transcripts | 2-3 min per episode |
| **Internet Required** | Yes | Yes |
| **Always Running** | No | Yes |

## Hybrid Approach (Best of Both Worlds)

### Local Content Generation
```bash
# On your local machine:
# 1. Generate threads with FREE YouTube transcripts
python main.py --dry-run --limit 5

# 2. Commit threads to git
git add output/threads/
git commit -m "Add daily podcast threads"
git push origin main
```

### Cloud Publishing
```bash
# On your cloud server:
# 1. Pull the new threads
git pull origin main

# 2. Post them to Twitter
DRY_RUN=false python post_existing_threads.py
```

## Troubleshooting Local Setup

### Issue: "Command not found: python3"
**Solution:**
```bash
# macOS: Install Python
brew install python3

# Windows: Download from python.org
# Or use: python instead of python3

# Linux (Ubuntu/Debian):
sudo apt update && sudo apt install python3 python3-pip python3-venv
```

### Issue: "No module named 'feedparser'"
**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: YouTube still blocked locally
**Possible causes:**
1. **Corporate network**: Some company networks block transcript API
2. **VPN**: Some VPNs use cloud provider IPs
3. **ISP restrictions**: Rare, but some ISPs might block it

**Solutions:**
```bash
# Test different video
python -c "from youtube_transcript_api import YouTubeTranscriptApi; print(YouTubeTranscriptApi().fetch('jNQXAC9IVRw', ['en']))"

# Try different network (mobile hotspot)
# Disable VPN if using one

# Fallback to Whisper locally
TRANSCRIPTION_METHODS=whisper python main.py --dry-run --limit 1
```

## Performance Optimization for Local Use

### Parallel Processing
```bash
# Process multiple podcasts simultaneously
python main.py --dry-run --limit 1 --podcast "Tim Ferriss" &
python main.py --dry-run --limit 1 --podcast "Lex Fridman" &
python main.py --dry-run --limit 1 --podcast "Naval Podcast" &
wait
echo "All threads generated!"
```

### Smart Caching
```bash
# Transcripts are automatically cached
ls .youtube_transcript_cache/  # See cached transcripts
ls .transcription_cache/      # See Whisper cache

# Clear cache if needed
rm -rf .youtube_transcript_cache .transcription_cache
```

## Next Steps After Local Setup

1. **Verify YouTube works**: `python -c "from youtube_transcript_api import YouTubeTranscriptApi; print('✅ Working!')"`
2. **Generate first thread**: `python main.py --dry-run --limit 1`
3. **Review output**: `cat output/threads/*.md`
4. **Setup X.com API**: Add credentials to .env for direct posting
5. **Create automation**: Set up daily/weekly content generation scripts
6. **Scale up**: Process multiple podcasts and accounts

## Ready to Start?

```bash
# Complete setup command:
git clone https://github.com/bluzername/AI-tweets.git && \
cd AI-tweets && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
cp .env.example .env && \
echo "✅ Setup complete! Edit .env with your OpenAI API key, then run:" && \
echo "python main.py --dry-run --limit 1"
```

**You'll get FREE YouTube transcription and save hundreds of dollars per month!** 🎉