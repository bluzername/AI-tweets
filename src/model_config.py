"""
Central model selection.

Every model id the pipeline sends to an API is looked up here by role, so a
retired or renamed model is a one-line .env change instead of a code hunt.
Values are read from the environment at call time (after python-dotenv has
loaded .env), falling back to the defaults below.

Role                    Env var                     Used by
----------------------  --------------------------  ----------------------------------
GPT_MODEL               GPT_MODEL                   analysis, thread generation, hooks
GPT_FAST_MODEL          GPT_FAST_MODEL              cheap helper calls, health checks
GPT_MINI_MODEL          GPT_MINI_MODEL              OSINT handle inference (direct OpenAI)
WHISPER_MODEL           WHISPER_MODEL               OpenAI transcription
CLAUDE_MODEL            CLAUDE_MODEL                multi-model consensus (direct)
GEMINI_MODEL            GEMINI_MODEL                multi-model consensus (direct)
OPENROUTER_GPT_MODEL    OPENROUTER_GPT_MODEL        multi-model consensus, viral pipeline
OPENROUTER_CLAUDE_MODEL OPENROUTER_CLAUDE_MODEL     multi-model consensus
OPENROUTER_GEMINI_MODEL OPENROUTER_GEMINI_MODEL     multi-model consensus
OPENROUTER_MINI_MODEL   OPENROUTER_MINI_MODEL       OSINT handle inference (OpenRouter)
OPENROUTER_HEALTH_MODEL OPENROUTER_HEALTH_MODEL     health check ping
"""

import os

DEFAULT_MODELS = {
    # OpenAI (still served as of 2026-09; whisper-1 is deprecated with a
    # Feb 2027 shutdown, gpt-transcribe is the announced replacement)
    "GPT_MODEL": "gpt-4-turbo-preview",
    "GPT_FAST_MODEL": "gpt-3.5-turbo",
    "GPT_MINI_MODEL": "gpt-4o-mini",
    "WHISPER_MODEL": "whisper-1",
    # Anthropic (claude-3-* ids are retired)
    "CLAUDE_MODEL": "claude-sonnet-5",
    # Google (gemini-pro / gemini-1.0-pro are gone from the model list)
    "GEMINI_MODEL": "gemini-2.5-flash",
    # OpenRouter slugs: check https://openrouter.ai/models and override in .env
    "OPENROUTER_GPT_MODEL": "openai/gpt-4-turbo-preview",
    "OPENROUTER_CLAUDE_MODEL": "anthropic/claude-3-sonnet",
    "OPENROUTER_GEMINI_MODEL": "google/gemini-pro-1.5",
    "OPENROUTER_MINI_MODEL": "openai/gpt-4o-mini",
    "OPENROUTER_HEALTH_MODEL": "meta-llama/llama-3.2-3b-instruct:free",
}


def get_model(role: str) -> str:
    """Model id for a role, from the environment or DEFAULT_MODELS."""
    if role not in DEFAULT_MODELS:
        raise KeyError(f"Unknown model role: {role}")
    value = os.getenv(role, "").strip()
    return value or DEFAULT_MODELS[role]
