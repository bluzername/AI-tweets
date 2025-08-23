#!/usr/bin/env python3
"""Simple test with a short audio file to validate pipeline."""

import logging
import os
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_test_config():
    """Create a minimal config for testing."""
    test_config = {
        "podcasts": [
            {
                "name": "Test Podcast",
                "rss_url": "https://www.thisamericanlife.org/podcast/rss.xml",  # Known working feed
                "x_handle": "ThisAmerLife",
                "creator_handles": ["iraglass"]
            }
        ],
        "accounts": [
            {
                "name": "main",
                "target_audience": "Test audience",
                "tone": "casual",
                "emoji_usage": "minimal"
            }
        ]
    }
    
    import json
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print("Created test_config.json")

def test_rss_parsing():
    """Test RSS parsing without transcription."""
    print("=== Testing RSS Parsing ===")
    
    from src.rss_parser import RSSFeedParser
    
    parser = RSSFeedParser(
        feed_urls=["https://www.thisamericanlife.org/podcast/rss.xml"],
        days_back=30
    )
    
    try:
        episodes = parser.parse_feeds()
        if episodes:
            print(f"✅ Found {len(episodes)} episodes")
            episode = episodes[0]
            print(f"Latest: {episode.title}")
            print(f"Audio URL: {episode.audio_url[:50]}...")
            print(f"YouTube URLs: {episode.youtube_urls}")
            return episode
        else:
            print("❌ No episodes found")
            return None
    except Exception as e:
        print(f"❌ RSS parsing failed: {e}")
        return None

def test_youtube_transcription():
    """Test YouTube transcription with a known video."""
    print("\n=== Testing YouTube Transcription ===")
    
    from src.youtube_transcriber import YouTubeTranscriber
    
    transcriber = YouTubeTranscriber()
    
    # Use a known video with transcripts (Rick Astley - Never Gonna Give You Up)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        video_id = transcriber.extract_video_id(test_url)
        print(f"Extracted video ID: {video_id}")
        
        if video_id:
            # This might fail due to cloud provider blocking, but test the code path
            result = transcriber.transcribe_from_url(test_url)
            print(f"✅ YouTube transcription successful: {len(result['text'])} chars")
            return True
    except Exception as e:
        print(f"⚠️  YouTube transcription failed (expected on cloud): {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n=== Testing Configuration ===")
    
    from src.config import Config
    
    try:
        config = Config(config_file="test_config.json")
        print(f"✅ Config loaded: {len(config.podcasts)} podcasts, {len(config.accounts)} accounts")
        print(f"Transcription methods: {config.transcription_methods}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_multi_transcriber():
    """Test multi-transcriber without actual API calls."""
    print("\n=== Testing MultiTranscriber ===")
    
    from src.multi_transcriber import MultiTranscriber
    
    try:
        transcriber = MultiTranscriber(
            openai_api_key="test_key",
            preferred_methods=["youtube", "whisper"]
        )
        
        stats = transcriber.get_transcription_stats()
        print(f"✅ MultiTranscriber initialized: {stats['methods_available']}")
        return True
    except Exception as e:
        print(f"❌ MultiTranscriber failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Simple Pipeline Test")
    print("=" * 50)
    
    # Create test config
    create_test_config()
    
    success = True
    
    # Test each component
    if not test_config_loading():
        success = False
    
    episode = test_rss_parsing()
    if not episode:
        success = False
    
    if not test_multi_transcriber():
        success = False
    
    # Optional YouTube test (may fail on cloud)
    test_youtube_transcription()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Basic tests passed! Pipeline components working.")
        print("\nNext steps:")
        print("1. Test with actual episode transcription:")
        print("   python main.py --config test_config.json --dry-run --limit 1")
        print("2. For large files, consider using YouTube-only mode:")
        print("   TRANSCRIPTION_METHODS=youtube python main.py --dry-run --limit 1")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    # Clean up
    if os.path.exists('test_config.json'):
        os.remove('test_config.json')
    
    return success

if __name__ == "__main__":
    main()