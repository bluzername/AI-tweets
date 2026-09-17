# AI-tweets

Turns podcast episodes into X (Twitter) threads and Telegram posts. It pulls episodes from RSS, gets a transcript (YouTube captions first, Whisper as fallback), asks an LLM for the most shareable insights, writes threads in configurable voices, and either posts them or writes them to Markdown.

## Entry points

| Command | What it runs |
|---------|--------------|
| `python main.py [--podcast NAME] [--limit N] [--dry-run] [--config config.json] [--env .env]` | The original one-shot pipeline: RSS to transcript to `AIAnalyzer` to `ThreadGenerator` to `MultiAccountPublisher`. Without X credentials it writes threads to `output/*.md`. |
| `python viral_main.py --mode {discovery,processing,scheduler,web,full} [--config viral_config.json] [--episode-limit N] [--web-port 5000] [--stats]` | The "viral" pipeline: episode database, viral insight extraction, tweet crafting, a scheduler/publisher with rate limiting and a Flask review UI. |
| `python daemon.py [--status]` | 24/7 orchestrator for Docker: discovery every `DISCOVERY_INTERVAL`, processing every `PROCESSING_INTERVAL`, posting every `SCHEDULER_CHECK_INTERVAL`, with health checks, auto-recovery, cost and tweet caps. |
| `python hebrew_orchestrator.py` | Hebrew podcast variant that publishes to Telegram only (`hebrew_config.json`). |
| `python src/web_dashboard.py --port 5000` | Read-only monitoring dashboard. |

Docker: `docker compose up -d` starts the daemon and the dashboard (`docker-compose.yml`); `docker-compose.gpu.yml` adds local Whisper on a GPU. A systemd unit lives in `deployment/`.

## Setup

```bash
pip install -r requirements.txt        # full pipeline (includes local Whisper, transformers, librosa)
cp .env.example .env                    # fill in keys
cp viral_config.example.json viral_config.json   # podcasts, accounts, schedule
python main.py --dry-run --limit 1
```

`main.py` creates `config.json` on first run if it is missing.

### Keys and configuration (`.env`)

- **LLM access**: either `USE_OPENROUTER=true` + `OPENROUTER_API_KEY` (one key for every model) or `OPENAI_API_KEY`, with optional `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` for the multi-model consensus analyzer.
- **X accounts**: `MAIN_*`, `CASUAL_*`, `PODDEBUNKER_*` API key, secret, access token, access token secret and bearer token. Missing credentials mean Markdown fallback instead of posting.
- **Fact checking**: `GOOGLE_FACTCHECK_API_KEY` and the `FACTCHECK_*` thresholds for the debunker account.
- **Daemon limits**: `MAX_EPISODES_PER_CYCLE`, `MAX_API_COST_PER_DAY`, `MAX_TWEETS_PER_DAY`, operating and quiet hours, alerting via SMTP, Slack or webhook.
- **Transcription**: `TRANSCRIPTION_METHODS=youtube,whisper,local_whisper` in preference order, `WHISPER_DEVICE=auto|cpu|cuda|mps`.

### Model ids

Every model id is a role resolved by `src/model_config.py` from the environment, so a retired model is a one-line `.env` change:

| Role | Default | Used for |
|------|---------|----------|
| `GPT_MODEL` | `gpt-4-turbo-preview` | insight extraction, thread generation, hooks |
| `GPT_FAST_MODEL` | `gpt-3.5-turbo` | cheap helper prompts, health check |
| `GPT_MINI_MODEL` | `gpt-4o-mini` | OSINT handle inference |
| `WHISPER_MODEL` | `whisper-1` | OpenAI transcription (deprecated by OpenAI, shutdown Feb 2027) |
| `CLAUDE_MODEL` | `claude-sonnet-5` | multi-model consensus (direct) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | multi-model consensus (direct, `google-genai` SDK) |
| `OPENROUTER_*_MODEL` | see `.env.example` | the same roles through OpenRouter; verify slugs at openrouter.ai/models |

## How a thread is made

```
RSS (feedparser) -> episode
  -> transcript: YouTube captions (youtube-transcript-api) | OpenAI Whisper | local Whisper
  -> insights: AIAnalyzer / ViralContentAnalyzer / MultiModelAnalyzer (consensus across providers)
  -> thread: ThreadGenerator (styles, hooks, emoji and hashtag optimisation, tweet splitting)
  -> quality gates: ContentValidator, ContentDeduplicator, FactChecker (PodDebunker account)
  -> publish: X via tweepy with X rate manager | Telegram | Markdown fallback
  -> tracking: cost tracker, performance tracker, A/B tests, feedback loop optimiser (SQLite in data/)
```

`src/thumbnail_fetcher` grabs episode art (Apple, RSS, YouTube, per-podcast scrapers) for image tweets; `src/osint_handle_finder` resolves guest X handles from show notes, Wikidata, web search and an LLM.

## Development

```bash
pip install -r requirements-test.txt   # light set, no torch
pytest                                 # tests/ (config in pytest.ini)
python -m compileall -q src main.py viral_main.py daemon.py hebrew_orchestrator.py scripts
```

`scripts/` holds manual smoke scripts (`test_openrouter.py`, `test_full_pipeline.py`, `demo_image_feature.py`, ...) that hit real APIs; they are not collected by pytest.

CI runs the same three commands on every push and pull request.

## Runtime layout

| Path | Contents |
|------|----------|
| `data/` | SQLite databases and JSON state created at runtime (git-ignored). Also holds `*_patched.py` copies of `viral_scheduler`, `autonomous_orchestrator`, `x_rate_manager` and the Telegram modules that `data/apply_rate_limit_patch.sh` copies over `src/` at container start. These have diverged from `src/` in both directions and still need a manual merge. |
| `logs/`, `output/`, `cache/`, `backups/` | Runtime, git-ignored, mounted as volumes in Docker. |
| `docs/` | Installation, local setup, deployment, autonomous deployment and cost optimisation guides. |

## Documentation

- [Installation](docs/INSTALLATION.md), [Local setup](docs/LOCAL_SETUP.md), [Setup guide](docs/SETUP_GUIDE.md)
- [Deployment](docs/DEPLOYMENT.md), [Autonomous deployment](docs/AUTONOMOUS_DEPLOYMENT.md), [deployment/](deployment/README.md)
- [Cost optimisation](docs/COST_OPTIMIZATION.md)
