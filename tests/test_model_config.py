"""Model role resolution."""

import pytest

from model_config import DEFAULT_MODELS, get_model


def test_defaults_are_used_when_env_unset(monkeypatch):
    monkeypatch.delenv("GPT_MODEL", raising=False)
    assert get_model("GPT_MODEL") == DEFAULT_MODELS["GPT_MODEL"]


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    assert get_model("CLAUDE_MODEL") == "claude-opus-5"


def test_blank_env_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "   ")
    assert get_model("GEMINI_MODEL") == "gemini-2.5-flash"


def test_unknown_role_is_an_error():
    with pytest.raises(KeyError):
        get_model("NOT_A_ROLE")


def test_no_retired_anthropic_or_gemini_defaults():
    for value in DEFAULT_MODELS.values():
        assert not value.startswith("claude-3-")
        assert value != "gemini-pro"
