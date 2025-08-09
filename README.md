# AI-tweets

An intelligent pipeline that converts podcast episodes into engaging X.com (Twitter) threads using AI.

## Features

- **RSS Feed Integration**: Automatically pulls recent episodes from configured podcast feeds
- **AI Transcription**: Uses OpenAI Whisper API for accurate audio transcription
- **Smart Analysis**: GPT-powered extraction of key insights and actionable takeaways
- **Thread Generation**: Creates compelling, platform-optimized threads with multiple style variations
- **Multi-Account Support**: Manages multiple X.com accounts with unique voices and audiences
- **Intelligent Fallback**: Saves threads to markdown when API access is unavailable
- **GIF Suggestions**: Recommends relevant, trendy GIFs for enhanced engagement
- **Robust Error Handling**: Comprehensive logging and graceful degradation

## Architecture

The pipeline follows a modular, extensible design:

```
RSS Feeds → Transcription → AI Analysis → Thread Generation → Publishing
```

### Core Modules

- `rss_parser.py`: Fetches and parses podcast RSS feeds
- `transcriber.py`: Handles audio transcription via Whisper API
- `ai_analyzer.py`: Extracts insights and scores highlights
- `thread_generator.py`: Creates platform-optimized threads
- `x_publisher.py`: Publishes to X.com with markdown fallback
- `config.py`: Manages configuration and credentials

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
- OpenAI API key (for Whisper and GPT)
- X.com API credentials (optional, for direct posting)

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
- `DAYS_BACK`: How many days to look back for episodes (default: 7)
- `MAX_HIGHLIGHTS`: Maximum highlights to extract (default: 3)
- `DRY_RUN`: Set to true to prevent posting (default: false)

### Account Styles

Each account can have unique:
- `tone`: witty, formal, casual, inspirational, educational
- `emoji_usage`: none, minimal, moderate, heavy
- `hashtag_style`: none, minimal, trending, custom
- `target_audience`: Specific audience description

## Future Enhancements

The architecture supports easy addition of:
- Highlight scoring algorithms
- Additional social platforms
- Audio snippet attachments
- Engagement analytics dashboard
- A/B testing for opening tweets
- Interactive polls and questions
- Multi-language support

## Contributing

Contributions are welcome! The modular architecture makes it easy to:
- Add new podcast sources
- Implement alternative AI providers
- Create new thread styles
- Add social platforms

## License

MIT

## Support

For issues or questions, please open an issue on GitHub.