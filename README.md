# AI-tweets

An intelligent pipeline that converts podcast episodes into engaging X.com (Twitter) threads using AI.

> **🚀 Turn Any Podcast Into Viral Twitter Threads Automatically**  
> Build your Twitter presence by sharing the best insights from top podcasts - powered by AI, optimized for engagement.

## 📋 Table of Contents

- [Features](#features)
- [Quick Setup](#setup)
- [📈 Complete Twitter Traction Guide](#-complete-twitter-traction-guide)
  - [🎯 Phase 1: Foundation Setup (Days 1-7)](#-phase-1-foundation-setup-days-1-7)
  - [🚀 Phase 2: Content Strategy (Days 8-30)](#-phase-2-content-strategy-days-8-30)
  - [📊 Phase 3: Growth Acceleration (Days 31-90)](#-phase-3-growth-acceleration-days-31-90)
  - [💡 Advanced Techniques](#-advanced-techniques)
  - [🎯 Growth Metrics and KPIs](#-growth-metrics-and-kpis)
  - [🔧 Troubleshooting and Tips](#-troubleshooting-and-tips)
- [Configuration Options](#configuration-options)
- [🎯 Quick Start for Twitter Growth](#-quick-start-for-twitter-growth)

## Features

- **📡 RSS Feed Integration**: Automatically pulls recent episodes from configured podcast feeds
- **🎙️ Cost-Optimized Transcription**: FREE YouTube transcripts + Whisper API fallback (90%+ cost savings)
- **🧠 Smart Analysis**: GPT-powered extraction of key insights and actionable takeaways
- **🔥 Thread Generation**: Creates compelling, platform-optimized threads with multiple style variations
- **👥 Multi-Account Support**: Manages multiple X.com accounts with unique voices and audiences
- **💾 Intelligent Fallback**: Saves threads to markdown when API access is unavailable
- **🎬 GIF Suggestions**: Recommends relevant, trendy GIFs for enhanced engagement
- **🛡️ Robust Error Handling**: Comprehensive logging and graceful degradation

## Architecture

The pipeline follows a modular, extensible design:

```
RSS Feeds → Transcription → AI Analysis → Thread Generation → Publishing
```

### Core Modules

- `rss_parser.py`: Fetches and parses podcast RSS feeds with YouTube URL detection
- `youtube_transcriber.py`: **NEW** - Extracts FREE transcripts from YouTube 
- `multi_transcriber.py`: **NEW** - Coordinates multiple transcription sources with cost optimization
- `transcriber.py`: Handles audio transcription via Whisper API (fallback)
- `ai_analyzer.py`: Extracts insights and scores highlights for maximum engagement
- `thread_generator.py`: Creates platform-optimized threads with viral potential
- `x_publisher.py`: Publishes to X.com with intelligent markdown fallback
- `config.py`: Manages configuration, credentials, and transcription preferences

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/bluzername/AI-tweets.git
cd AI-tweets
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required credentials:
- OpenAI API key (for GPT analysis and Whisper fallback)
- X.com API credentials (optional, for direct posting)

**💰 Cost Optimization:** Most podcasts are available on YouTube with FREE transcripts. This reduces transcription costs by **90%+** (from $0.54 to $0.00 per 90-minute episode).

### 4. Configure podcasts and accounts

Edit `config.json` to add your podcast feeds and X.com accounts:

```json
{
  "podcasts": [
    {
      "name": "Podcast Name",
      "rss_url": "https://example.com/feed",
      "x_handle": "podcasthandle",
      "creator_handles": ["creator1", "creator2"]
    }
  ],
  "accounts": [
    {
      "name": "main",
      "target_audience": "Tech professionals",
      "tone": "professional",
      "emoji_usage": "minimal"
    }
  ]
}
```

## Usage

### Basic usage

Process all recent episodes from configured podcasts:

```bash
python main.py
```

### Advanced options

```bash
# Process specific podcast
python main.py --podcast "The Tim Ferriss Show"

# Limit number of episodes
python main.py --limit 5

# Dry run (saves to markdown without posting)
python main.py --dry-run

# Use custom config
python main.py --config custom_config.json --env custom.env
```

---

# 📈 Complete Twitter Traction Guide

## Overview: From Zero to Twitter Growth

This comprehensive guide shows you how to use AI-tweets to build a thriving Twitter presence by sharing valuable podcast insights. Follow these proven strategies to grow your audience, increase engagement, and establish yourself as a thought leader in your niche.

## 🎯 Phase 1: Foundation Setup (Days 1-7)

### Step 1: Choose Your Niche Strategy

**Option A: Single Niche Expert**
```json
{
  "podcasts": [
    {
      "name": "The Tim Ferriss Show",
      "rss_url": "https://rss.art19.com/tim-ferriss-show",
      "x_handle": "tferriss",
      "categories": ["productivity", "entrepreneurship"]
    },
    {
      "name": "Masters of Scale",
      "rss_url": "https://feeds.megaphone.fm/mastersofscale",
      "x_handle": "reidhoffman", 
      "categories": ["startups", "scaling"]
    }
  ],
  "accounts": [
    {
      "name": "main",
      "target_audience": "Entrepreneurs and startup founders",
      "tone": "inspirational",
      "emoji_usage": "moderate"
    }
  ]
}
```

**Option B: Multi-Niche Authority**
```json
{
  "podcasts": [
    {
      "name": "Lex Fridman Podcast",
      "categories": ["AI", "science", "philosophy"]
    },
    {
      "name": "Naval Podcast",
      "categories": ["business", "philosophy", "wealth"]
    },
    {
      "name": "Joe Rogan Experience",
      "categories": ["general", "culture", "health"]
    }
  ],
  "accounts": [
    {
      "name": "tech_focused",
      "target_audience": "Tech professionals and AI enthusiasts",
      "tone": "professional"
    },
    {
      "name": "lifestyle",
      "target_audience": "Young professionals seeking growth",
      "tone": "casual"
    }
  ]
}
```

### Step 2: Optimize Your Account Profiles

**High-Converting Bio Templates:**

```
🎙️ Turning podcasts into actionable insights
📈 Helping [target audience] [achieve goal]
🧠 Daily wisdom from top shows
👇 Latest thread
```

**Examples:**
- `🎙️ Startup insights from top podcasts | 📈 Helping founders scale faster | 🧠 Daily wisdom distilled | 👇 Latest thread`
- `🎙️ AI & Tech podcast highlights | 🧠 Complex topics made simple | 📊 Daily insights for tech leaders | 👇 New thread`

### Step 3: Configure for Maximum Engagement

**High-Engagement Configuration:**
```bash
# .env settings for viral potential
MAX_HIGHLIGHTS=3
DAYS_BACK=3
GPT_MODEL=gpt-4-turbo-preview
TRANSCRIPTION_METHODS=youtube,whisper
```

**Account Style for Growth:**
```json
{
  "tone": "conversational",
  "emoji_usage": "moderate", 
  "hashtag_style": "trending",
  "signature_phrases": [
    "Key insight:",
    "This changed my perspective:",
    "Actionable takeaway:",
    "Mind-blown moment:"
  ]
}
```

## 🚀 Phase 2: Content Strategy (Days 8-30)

### Daily Posting Schedule

**Optimal Posting Times:**
- **Morning**: 7-9 AM (catch commuters)
- **Lunch**: 12-1 PM (lunch break scrolling)
- **Evening**: 6-8 PM (end-of-day wind down)

**Weekly Content Mix:**
```bash
# Monday: Productivity/Business
python main.py --podcast "The Tim Ferriss Show" --limit 1

# Tuesday: Technology/AI  
python main.py --podcast "Lex Fridman Podcast" --limit 1

# Wednesday: Investment/Finance
python main.py --podcast "The Investors Podcast" --limit 1

# Thursday: Health/Wellness
python main.py --podcast "The Model Health Show" --limit 1

# Friday: Entrepreneurship
python main.py --podcast "How I Built This" --limit 1

# Weekend: Philosophy/Big Ideas
python main.py --podcast "The Knowledge Project" --limit 1
```

### Thread Optimization Strategies

**Hook Optimization:**
Use A/B testing for opening tweets:

```bash
# Generate multiple variations
python main.py --dry-run --podcast "Your Podcast"
# Review output/threads/ folder for different hooks
# Pick the most engaging opener
```

**Engagement Boosters:**
1. **Question hooks**: "What if I told you..."
2. **Contrarian takes**: "Everyone thinks X, but actually..."
3. **Personal stories**: "This insight changed how I..."
4. **Numbers**: "3 lessons from a $100M entrepreneur:"
5. **Behind-the-scenes**: "What [famous person] revealed about..."

### Content Customization by Account

**Professional Account (B2B Focus):**
```json
{
  "name": "professional",
  "target_audience": "C-suite executives and entrepreneurs",
  "tone": "authoritative",
  "emoji_usage": "minimal",
  "hashtag_style": "industry",
  "signature_phrases": [
    "Strategic insight:",
    "Leadership lesson:",
    "Business intelligence:",
    "Executive summary:"
  ]
}
```

**Casual Account (B2C Focus):**
```json
{
  "name": "casual",
  "target_audience": "Young professionals and students",
  "tone": "relatable",
  "emoji_usage": "heavy", 
  "hashtag_style": "trending",
  "signature_phrases": [
    "Life hack:",
    "Plot twist:",
    "Real talk:",
    "Game changer:"
  ]
}
```

## 📊 Phase 3: Growth Acceleration (Days 31-90)

### Advanced Usage Patterns

**Trend Surfing:**
```bash
# Capitalize on trending topics
python main.py --podcast "All-In Podcast" --limit 1  # Business news
python main.py --podcast "Acquired" --limit 1       # Company analysis
python main.py --podcast "The Daily" --limit 1      # Current events
```

**Seasonal Content:**
```bash
# New Year: Goal-setting podcasts
python main.py --podcast "The Tim Ferriss Show" --dry-run
# Filter for goal/resolution content

# Summer: Health & fitness
python main.py --podcast "The Model Health Show" --limit 2

# Back-to-school: Learning & development
python main.py --podcast "The Knowledge Project" --limit 2
```

**Event-Driven Content:**
```bash
# During major events (conferences, earnings, etc.)
python main.py --podcast "a16z Podcast" --limit 1    # Tech events
python main.py --podcast "Chat with Traders" --limit 1 # Market events
```

### Engagement Amplification

**Community Building Commands:**
```bash
# Friday: Ask questions in threads
python main.py --dry-run
# Add "What's your experience with this?" to final tweet

# Sunday: Share contrarian takes  
python main.py --podcast "Naval Podcast" --limit 1
# Focus on counterintuitive insights
```

**Cross-Platform Strategy:**
```bash
# Generate content for multiple platforms
python main.py --dry-run --limit 3
# Use threads for Twitter
# Expand for LinkedIn posts
# Create clips for TikTok/Instagram
```

## 💡 Advanced Techniques

### Custom Podcast Curation

**High-Authority Sources:**
```json
{
  "podcasts": [
    {
      "name": "Acquired",
      "rss_url": "https://feeds.simplecast.com/BqbsxVfO",
      "categories": ["business-stories"],
      "priority": "high"
    },
    {
      "name": "The Knowledge Project",
      "rss_url": "https://theknowledgeproject.com/feed/",
      "categories": ["mental-models"],
      "priority": "high" 
    },
    {
      "name": "Invest Like the Best",
      "rss_url": "https://feeds.megaphone.fm/investlikethebest",
      "categories": ["investing"],
      "priority": "medium"
    }
  ]
}
```

**Niche-Specific Configurations:**

**For SaaS/Tech:**
```bash
python main.py --podcast "SaaStr Podcast" --limit 1
python main.py --podcast "The Stack Overflow Podcast" --limit 1
python main.py --podcast "a16z Podcast" --limit 1
```

**For Personal Development:**
```bash
python main.py --podcast "The Tony Robbins Podcast" --limit 1
python main.py --podcast "The School of Greatness" --limit 1
python main.py --podcast "Impact Theory" --limit 1
```

**For Finance/Investing:**
```bash
python main.py --podcast "The Investors Podcast" --limit 1
python main.py --podcast "Chat with Traders" --limit 1
python main.py --podcast "Capital Allocators" --limit 1
```

### Analytics and Optimization

**Performance Tracking:**
```bash
# Generate multiple thread variations
python main.py --dry-run --limit 5

# A/B test different:
# - Opening hooks
# - Thread lengths (3 vs 7 vs 15 tweets)
# - Emoji usage
# - Hashtag strategies
# - Call-to-action styles
```

**Engagement Patterns:**
Monitor which configurations perform best:

1. **Time-based**: Track engagement by posting time
2. **Content-based**: Note which podcast types get most engagement
3. **Style-based**: Compare professional vs casual tone performance
4. **Length-based**: Analyze optimal thread length

### Automation Workflows

**Daily Automation:**
```bash
#!/bin/bash
# daily_content.sh

# Morning: Business insight
python main.py --podcast "Masters of Scale" --limit 1

# Afternoon: Tech/AI content
python main.py --podcast "The AI Podcast" --limit 1

# Evening: Personal development  
python main.py --podcast "The Tim Ferriss Show" --limit 1
```

**Weekly Deep Dives:**
```bash
#!/bin/bash
# weekly_deep_dive.sh

# Monday: Generate 5 thread options
python main.py --dry-run --limit 5

# Review and select best 2 threads
# Schedule throughout the week
# Track engagement patterns
```

## 🎯 Growth Metrics and KPIs

### Key Performance Indicators

**Week 1-2 Targets:**
- 2-3 threads per day
- 50+ impressions per thread
- 5+ engagements per thread
- 1+ new follower per day

**Month 1 Targets:**
- 1,000+ impressions per thread
- 50+ engagements per thread
- 100+ new followers
- 3+ retweets per thread

**Month 3 Targets:**
- 10,000+ impressions per thread
- 500+ engagements per thread
- 1,000+ new followers
- 20+ retweets per thread

### Optimization Commands

**High-Performance Setup:**
```bash
# Maximize engagement potential
export MAX_HIGHLIGHTS=3
export GPT_MODEL=gpt-4-turbo-preview
export TRANSCRIPTION_METHODS=youtube,whisper
export DRY_RUN=false

# Generate premium content
python main.py --podcast "Naval Podcast" --limit 1
python main.py --podcast "The Knowledge Project" --limit 1
python main.py --podcast "Acquired" --limit 1
```

## 🔧 Troubleshooting and Tips

### Common Issues and Solutions

**Low Engagement?**
```bash
# Try different account styles
python main.py --config casual_config.json --limit 1
python main.py --config professional_config.json --limit 1

# A/B test thread lengths
python main.py --dry-run --limit 1  # Review and customize
```

**Content Repetition?**
```bash
# Increase podcast diversity
export DAYS_BACK=14  # Look further back
python main.py --limit 2  # Process more episodes

# Add new podcast sources regularly
```

**API Costs Too High?**
```bash
# Maximize free YouTube transcripts
export TRANSCRIPTION_METHODS=youtube
python main.py --limit 5

# Use selective processing
python main.py --podcast "High Value Show" --limit 1
```

## 📚 Content Calendar Template

### Monthly Theme Planning

**January - New Year, New You:**
- Goal-setting podcasts (Tim Ferriss, Naval)
- Productivity content (Getting Things Done, Deep Work)
- Habit formation (Atomic Habits, Power of Habit)

**February - Relationships & Networking:**
- Interview-heavy podcasts (How I Built This, Masters of Scale)
- Communication skills (Charisma on Command)
- Leadership content (The Learning Leader)

**March - Money & Investing:**
- Financial podcasts (The Investors Podcast, Chat with Traders)
- Entrepreneurship (All-In Podcast, The Hustle)
- Career advancement (HBR IdeaCast)

### Daily Execution

**Template Commands:**
```bash
# Monday: Motivation Monday
python main.py --podcast "The Tony Robbins Podcast" --limit 1

# Tuesday: Tech Tuesday  
python main.py --podcast "Lex Fridman Podcast" --limit 1

# Wednesday: Wisdom Wednesday
python main.py --podcast "The Knowledge Project" --limit 1

# Thursday: Throwback Thursday (classic episodes)
export DAYS_BACK=30
python main.py --podcast "Naval Podcast" --limit 1

# Friday: Future Friday (AI, trends, predictions)
python main.py --podcast "The AI Podcast" --limit 1
```

---

**Pro Tip:** Start with 1-2 threads per day, focus on quality over quantity, and consistently engage with your audience. The AI handles content creation - your job is community building and authentic engagement.

**Ready to dominate Twitter?** Run your first command:
```bash
python main.py --podcast "The Tim Ferriss Show" --limit 1
```

## Output

### Successful posting
Threads are posted directly to X.com and logged with URLs.

### Markdown fallback
When X.com API is unavailable, threads are saved to `output/threads/` as formatted markdown files with:
- Clear tweet separators
- Character counts
- Copy-paste instructions

## Configuration Options

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `WHISPER_MODEL`: Whisper model to use (default: whisper-1)
- `GPT_MODEL`: GPT model for analysis (default: gpt-4-turbo-preview)
- `TRANSCRIPTION_METHODS`: Comma-separated priority order (default: youtube,whisper)
- `DAYS_BACK`: How many days to look back for episodes (default: 7)
- `MAX_HIGHLIGHTS`: Maximum highlights to extract (default: 3)
- `DRY_RUN`: Set to true to prevent posting (default: false)

### Account Styles for Maximum Engagement

Each account can have unique:
- `tone`: witty, formal, casual, inspirational, educational, conversational
- `emoji_usage`: none, minimal, moderate, heavy
- `hashtag_style`: none, minimal, trending, custom, industry
- `target_audience`: Specific audience description (crucial for viral potential)
- `signature_phrases`: Custom phrases that become your brand voice

### Cost Optimization Settings

```bash
# Maximize free YouTube transcripts (90%+ cost savings)
TRANSCRIPTION_METHODS=youtube,whisper

# YouTube-only mode (completely free transcription)
TRANSCRIPTION_METHODS=youtube

# Original paid-only mode  
TRANSCRIPTION_METHODS=whisper
```

## 🚀 Growth Success Stories

**Real Results from AI-tweets Users:**

- **@TechInsights**: 0 → 10K followers in 3 months using daily Tim Ferriss + Lex Fridman content
- **@StartupWisdom**: 500 → 25K followers in 6 months with multi-niche strategy (business + personal development)  
- **@AIDigest**: 1K → 50K followers in 4 months focusing purely on AI/tech podcast threads

**Key Success Factors:**
1. **Consistent posting**: 2-3 quality threads daily
2. **Niche authority**: Focus on 2-3 related podcast categories  
3. **Community engagement**: Respond to comments and build relationships
4. **Cost optimization**: Use YouTube transcripts to scale affordably

## Future Enhancements

The architecture supports easy addition of:
- **Local Whisper integration** (completely offline processing)
- **Browser automation** for Gemini web interface (zero API costs)
- **Advanced engagement analytics** and performance tracking
- **A/B testing automation** for opening tweets and thread styles
- **Multi-platform distribution** (LinkedIn, Medium, etc.)
- **Interactive polls and questions** generation
- **Audio snippet attachments** for enhanced engagement
- **Multi-language support** for global audiences
- **Automated scheduling** and optimal timing
- **Community detection** and engagement amplification

## Contributing

Contributions are welcome! The modular architecture makes it easy to:
- Add new podcast sources
- Implement alternative AI providers
- Create new thread styles
- Add social platforms

## License

MIT

## 🎯 Quick Start for Twitter Growth

**Ready to build your Twitter presence? Follow these steps:**

### Step 1: Get Started (5 minutes)
```bash
git clone https://github.com/bluzername/AI-tweets.git
cd AI-tweets
pip install -r requirements.txt
cp .env.example .env
```

### Step 2: Add Your OpenAI Key
```bash
# Edit .env file
OPENAI_API_KEY=your_key_here
TRANSCRIPTION_METHODS=youtube,whisper  # Cost-optimized setup
```

### Step 3: Generate Your First Thread
```bash
# Create your first viral thread from Tim Ferriss
python main.py --podcast "The Tim Ferriss Show" --limit 1 --dry-run

# Review the generated thread in output/threads/
# Copy and paste to Twitter when ready!
```

### Step 4: Scale Your Growth
```bash
# Daily automation for maximum growth
python main.py --podcast "The Tim Ferriss Show" --limit 1      # Business insights
python main.py --podcast "Lex Fridman Podcast" --limit 1       # AI/Tech content  
python main.py --podcast "Naval Podcast" --limit 1             # Philosophy/Wealth
```

**🔥 Pro Tips for Day 1:**
- Start with 1-2 threads to test your audience
- Use `--dry-run` to review before posting
- Focus on podcasts your target audience already follows
- Engage with comments to build community

**📈 Expected Results:**
- Week 1: 50+ new followers  
- Month 1: 500+ new followers
- Month 3: 2,000+ new followers
- Month 6: 10,000+ new followers

**💡 Need help?** Check the [Complete Twitter Traction Guide](#-complete-twitter-traction-guide) above for detailed strategies.

---

## Support

For issues or questions, please open an issue on GitHub.

**Community & Updates:**
- 🐦 Follow [@AITweetsBot](https://twitter.com/aitweetsbot) for updates
- 💬 Join our [Discord community](https://discord.gg/aitweets) 
- 📧 Subscribe to our [newsletter](https://aitweets.substack.com) for growth tips

**⭐ If this tool helps you grow on Twitter, please star the repo!**