# Installation & Setup Guide

## Quick Start (5 minutes)

### 1. Clone and Setup
```bash
git clone https://github.com/bluzername/AI-tweets.git
cd AI-tweets

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your OpenAI API key
nano .env  # Or use your preferred editor
```

**Minimum required in .env:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
TRANSCRIPTION_METHODS=youtube,whisper
DRY_RUN=true
```

### 3. Test Installation
```bash
# Run pipeline test
python test_full_pipeline.py

# Generate your first thread (dry-run mode)
python main.py --dry-run --limit 1
```

If test passes, you're ready to go! 🎉

## Detailed Setup

### Environment Variables (.env file)

```bash
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here
WHISPER_MODEL=whisper-1
GPT_MODEL=gpt-4o-mini

# Cost Optimization (RECOMMENDED)
TRANSCRIPTION_METHODS=youtube,whisper  # Try free YouTube first

# Pipeline Configuration
DAYS_BACK=7          # How far back to look for episodes
MAX_HIGHLIGHTS=3     # Number of key insights to extract
DRY_RUN=true        # Save to markdown instead of posting

# X.com API Credentials (OPTIONAL - for direct posting)
MAIN_API_KEY=your_x_api_key
MAIN_API_SECRET=your_x_api_secret
MAIN_ACCESS_TOKEN=your_x_access_token
MAIN_ACCESS_TOKEN_SECRET=your_x_access_token_secret
MAIN_BEARER_TOKEN=your_x_bearer_token
```

### Podcast Configuration (config.json)

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
      "target_audience": "Tech professionals and entrepreneurs",
      "tone": "professional",
      "emoji_usage": "minimal",
      "hashtag_style": "trending"
    }
  ]
}
```

## Common Issues & Solutions

### Issue: "No module named 'feedparser'"
**Solution:** Activate virtual environment first
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Audio file too large for Whisper API"
**Solutions:**
1. **Use YouTube transcripts (FREE):**
   ```bash
   TRANSCRIPTION_METHODS=youtube,whisper
   ```
2. **YouTube-only mode:**
   ```bash
   TRANSCRIPTION_METHODS=youtube
   ```
3. **Skip large files:**
   ```bash
   python main.py --limit 1  # Process only recent episodes
   ```

### Issue: YouTube API blocked (cloud providers)
**This is expected behavior.** YouTube blocks most cloud IPs.

**Solutions:**
1. **Local development:** YouTube API works on home/office networks
2. **Cloud deployment:** Use Whisper-only mode:
   ```bash
   TRANSCRIPTION_METHODS=whisper
   ```
3. **Hybrid approach:** Process on local machine, deploy threads

### Issue: OpenAI API errors
**Solutions:**
1. **Invalid API key:** Check your key at [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Rate limiting:** Add delays between requests:
   ```bash
   python main.py --limit 1  # Process one at a time
   ```
3. **Cost management:** Use cost-optimized settings:
   ```bash
   TRANSCRIPTION_METHODS=youtube,whisper
   MAX_HIGHLIGHTS=2
   ```

### Issue: RSS feed parsing errors
**Common causes:**
- Outdated RSS URLs (404 errors)
- Feeds returning HTML instead of XML
- Network connectivity issues

**Solutions:**
1. **Test individual feeds:**
   ```bash
   curl -I "https://your-rss-feed-url"
   ```
2. **Use working feeds:** Check our [recommended podcasts list](#recommended-podcasts)
3. **Update config.json** with current URLs

### Issue: Python 3.13 compatibility
**Known limitation:** pydub has issues with Python 3.13 due to missing `audioop` module.

**Solutions:**
1. **Use Python 3.11 or 3.12:**
   ```bash
   pyenv install 3.12.0
   pyenv local 3.12.0
   ```
2. **Skip large files:** Pipeline handles this automatically
3. **Use YouTube transcripts** to avoid audio processing

## Recommended Podcasts

These podcasts have reliable RSS feeds and often include YouTube versions:

### Business & Entrepreneurship
```json
{
  "name": "The Tim Ferriss Show",
  "rss_url": "https://rss.art19.com/tim-ferriss-show"
},
{
  "name": "Masters of Scale",
  "rss_url": "https://feeds.megaphone.fm/mastersofscale"
},
{
  "name": "How I Built This",
  "rss_url": "https://feeds.npr.org/510313/podcast.xml"
}
```

### Technology & AI
```json
{
  "name": "Lex Fridman Podcast",
  "rss_url": "https://lexfridman.com/feed/podcast/"
},
{
  "name": "a16z Podcast", 
  "rss_url": "https://feeds.soundcloud.com/users/soundcloud:users:7398766/sounds.rss"
},
{
  "name": "This Week in Tech",
  "rss_url": "https://feeds.twit.tv/twit.xml"
}
```

### Personal Development
```json
{
  "name": "The Knowledge Project",
  "rss_url": "https://theknowledgeproject.com/feed/"
},
{
  "name": "Naval Podcast",
  "rss_url": "https://podcasts.apple.com/us/podcast/naval/id1454097755"
}
```

## Testing Your Setup

### 1. Component Test
```bash
python test_full_pipeline.py
```
Should output: `🎉 PIPELINE TEST PASSED!`

### 2. RSS Parsing Test
```bash
python -c "
from src.rss_parser import RSSFeedParser
parser = RSSFeedParser(['https://rss.art19.com/tim-ferriss-show'], days_back=7)
episodes = parser.parse_feeds()
print(f'Found {len(episodes)} episodes')
if episodes:
    print(f'Latest: {episodes[0].title}')
"
```

### 3. End-to-End Test (Dry Run)
```bash
python main.py --dry-run --limit 1 --podcast "The Tim Ferriss Show"
```

Check `output/threads/` for generated markdown files.

### 4. Production Test (with posting)
```bash
# Only after adding X.com API credentials
DRY_RUN=false python main.py --limit 1
```

## Performance Optimization

### Cost Optimization
- **Use YouTube transcripts:** `TRANSCRIPTION_METHODS=youtube,whisper`
- **Limit highlights:** `MAX_HIGHLIGHTS=2`
- **Process selectively:** `python main.py --podcast "Specific Show" --limit 1`

### Speed Optimization
- **Parallel processing:** Process multiple podcasts simultaneously
- **Cache utilization:** Transcripts are automatically cached
- **Smaller episodes:** Focus on podcasts with <30 minute episodes

### Quality Optimization
- **Target audience:** Be specific in account configuration
- **Tone matching:** Match account tone to your brand voice
- **Content filtering:** Choose high-quality, business-focused podcasts

## Next Steps

Once installed successfully:

1. **Read the [Complete Twitter Traction Guide](README.md#-complete-twitter-traction-guide)**
2. **Configure your first niche** with 2-3 related podcasts  
3. **Generate your first 5 threads** in dry-run mode
4. **Set up X.com API credentials** for direct posting
5. **Create automation scripts** for daily content generation

## Support

- 🐛 **Issues:** [GitHub Issues](https://github.com/bluzername/AI-tweets/issues)
- 📚 **Documentation:** [Complete Guide](README.md)
- 💬 **Community:** [Discord Server](https://discord.gg/aitweets)

---

**Ready to dominate Twitter?** Run your first command:
```bash
python main.py --dry-run --limit 1
```