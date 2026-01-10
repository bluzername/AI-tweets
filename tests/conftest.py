"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
import os
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Mock OpenAI API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")


@pytest.fixture
def mock_anthropic_key(monkeypatch):
    """Mock Anthropic API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")


@pytest.fixture
def mock_google_key(monkeypatch):
    """Mock Google AI API key."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-12345")


@pytest.fixture
def sample_transcript():
    """Sample podcast transcript for testing."""
    return """
    Welcome to the startup podcast. Today we're talking about the secret to success.

    The truth is, 90% of startups fail not because of bad ideas, but because they
    run out of money before finding product-market fit. The average time to PMF is
    18 months, but most founders only have runway for 12 months.

    Here's what successful founders do differently: They focus obsessively on one
    metric and ignore everything else. They talk to users every single day. And they're
    willing to pivot when the data tells them to.

    Remember: Speed beats perfection. Launch fast, learn faster.
    """


@pytest.fixture
def sample_insight():
    """Sample insight for testing."""
    return {
        "text": "90% of startups fail not because of bad ideas, but because they run out of money before finding product-market fit.",
        "category": "fact",
        "viral_score": 0.85,
        "confidence": 0.90
    }


@pytest.fixture
def sample_tweet():
    """Sample tweet for testing."""
    return {
        "text": "Ever wonder why startups fail?\n\n90% run out of money before finding product-market fit. The average time to PMF is 18 months, but most founders only have 12 months of runway.\n\nSpeed beats perfection. Launch fast, learn faster. 🚀\n\n#Startup #Entrepreneurship",
        "tweet_id": "123456789",
        "posted_at": "2025-01-19T12:00:00"
    }
