# Claude Code Implementation Rules for Podcast Highlight Thread MVP

## Project Overview
- Build a modular, scalable Python MVP that:
  - Pulls recent episodes from multiple podcast RSS feeds.
  - Transcribes audio using OpenAI Whisper API (secured via `.env`).
  - Uses GPT-5 to extract the top 2-3 insightful, actionable highlights.
  - Crafts engaging, multi-tweet X.com threads (with dynamic teasers and thread storytelling).
  - Supports multiple X.com accounts with distinct audience profiles and tone.
  - Posts threads via X.com API or falls back to `.md` file output if API keys are missing or invalid.
  - Logs all key actions, errors, and decisions for debugging and analytics.

## Core Functional Requirements
- Fetch latest episodes from configurable RSS feeds.
- Transcribe audio with OpenAI Whisper API (API key read securely from `.env`).
- Use GPT-5 to:
  - Extract 2–3 original, compelling highlights per episode.
  - Generate multi-tweet X.com threads including:
    - Context-setting first tweet with multiple dynamic teaser options for A/B testing.
    - Follow-up tweets explaining highlights with actionable takeaways.
    - Final tweet tagging podcast creators and including a strong call to action.
    - Suggest a cool, fresh, contextually relevant GIF.
- Support multiple X.com accounts, each with unique API keys, audience, style, and tone.
- Graceful fallback: If posting fails or lacks valid credentials, generate a clear, copy-paste ready `.md` file with thread splits (`---` or `[Thread Break]`).
- Extensive logging with timestamps and error reporting.

## Code Quality Guidelines
- Write clean, modular, well-documented Python code.
- Use modern packages: `requests`, `dotenv`, `openai`, etc.
- Include concise docstrings and inline comments explaining design choices.
- Handle API limits, edge cases, and errors thoughtfully.
- Design for extensibility: easy to add more podcasts, languages, or social platforms.

## Output Formatting
- Compose tweets in an engaging, human tone with emojis and hashtags where fitting.
- `.md` fallback files should:
  - Separate tweets clearly for easy copying and pasting.
  - Include thread openings with teaser options.
  - End threads with creator tags and calls to action.

## Bonus Ideas & Future Enhancements
- Highlight scoring based on “tweet-ability” and engagement potential.
- Personality/tone switching per X.com account.
- Incorporate user polls or questions in final tweets.
- Dashboard for monitoring thread impact and engagement.
- Include multimedia such as audio snippets alongside threads.

## Developer Workflow Tips
- Use `/clear` often to keep context focused.
- Provide detailed instructions upfront to maximize Claude’s success rate.
- Use multiple Claude instances or subagents for review and iteration.
- Document all environment setup steps (including `.env` handling for OpenAI and X.com keys).
- Write tests to cover fetching, transcription, GPT-5 prompt handling, posting, and fallback logic.
- Commit early and often with clear messages about features and fixes.

## Bash/CLI Commands (examples)
- `python main.py` — Run the main MVP script.
- `pip install -r requirements.txt` — Install dependencies.
- `cp .env.example .env` — Setup environment file with API keys.
- `python test_suite.py` — Run automated tests.

***

This file will automatically be pulled into Claude Code sessions, providing clear guidance to ensure high-quality, robust implementations aligned with your vision for the podcast highlight thread MVP.

If you want, I can help you draft sample tests, environmental setup instructions, or example `.env` files next.
