#!/usr/bin/env python3
"""Test script for YouTube transcription functionality."""

import logging
import os
from dotenv import load_dotenv

from src.youtube_transcriber import YouTubeTranscriber, find_youtube_urls
from src.multi_transcriber import MultiTranscriber

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_youtube_url_extraction():
    """Test YouTube URL extraction from text."""
    test_texts = [
        "Check out this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "Also on YouTube: https://youtu.be/dQw4w9WgXcQ",
        "Embedded link: https://youtube.com/embed/dQw4w9WgXcQ",
        "Multiple URLs: https://youtu.be/abc123 and https://www.youtube.com/watch?v=def456",
        "No YouTube URLs here, just some text about podcasts."
    ]
    
    print("=== Testing YouTube URL Extraction ===")
    for i, text in enumerate(test_texts, 1):
        urls = find_youtube_urls(text)
        print(f"{i}. Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"   Found URLs: {urls}")
        print()

def test_video_id_extraction():
    """Test video ID extraction from various YouTube URL formats."""
    transcriber = YouTubeTranscriber()
    
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=123s",
        "https://www.youtube.com/v/dQw4w9WgXcQ",
        "not_a_youtube_url.com",
        ""
    ]
    
    print("=== Testing Video ID Extraction ===")
    for url in test_urls:
        video_id = transcriber.extract_video_id(url)
        print(f"URL: {url}")
        print(f"Video ID: {video_id}")
        print()

def test_transcript_availability():
    """Test checking transcript availability for known videos."""
    transcriber = YouTubeTranscriber()
    
    # Use a known public video with transcripts (Rick Roll - should have auto captions)
    test_video_id = "dQw4w9WgXcQ"
    
    print("=== Testing Transcript Availability ===")
    try:
        available = transcriber.get_available_transcripts(test_video_id)
        print(f"Available transcripts for video {test_video_id}:")
        for transcript in available:
            print(f"  - {transcript['language_name']} ({transcript['language_code']})")
            print(f"    Generated: {transcript['is_generated']}, Translatable: {transcript['is_translatable']}")
        print()
        
        # Try to get a transcript
        if available:
            print("Attempting to fetch transcript...")
            result = transcriber.get_transcript(test_video_id)
            print(f"Success! Got {len(result['text'])} characters of transcript")
            print(f"Language: {result['language']} ({result['language_name']})")
            print(f"Duration: {result.get('duration', 'unknown')} seconds")
            print(f"Type: {result['transcript_type']}")
            print(f"Sample text: {result['text'][:200]}...")
        else:
            print("No transcripts available for this video")
            
    except Exception as e:
        print(f"Error testing transcript: {e}")

def test_multi_transcriber():
    """Test the MultiTranscriber with different configurations."""
    load_dotenv()
    
    print("=== Testing MultiTranscriber ===")
    
    # Test with YouTube-first priority
    transcriber = MultiTranscriber(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        preferred_methods=["youtube", "whisper"]
    )
    
    print("Configuration:")
    stats = transcriber.get_transcription_stats()
    print(f"Available methods: {stats['methods_available']}")
    print(f"Preferred order: {stats['preferred_order']}")
    print(f"YouTube enabled: {stats['youtube_enabled']}")
    print(f"Whisper enabled: {stats['whisper_enabled']}")
    print()
    
    # Test transcription with a known video
    test_youtube_urls = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    fake_audio_url = "https://example.com/fake_audio.mp3"
    
    try:
        print("Testing episode transcription...")
        result = transcriber.transcribe_episode(
            audio_url=fake_audio_url,
            youtube_urls=test_youtube_urls,
            title="Test Episode"
        )
        
        if result.get('text'):
            print(f"✅ Success! Method used: {result.get('method_used')}")
            print(f"Got {len(result['text'])} characters")
            print(f"Sample: {result['text'][:150]}...")
        else:
            print(f"❌ Failed. Error: {result.get('error')}")
            print(f"Attempts: {result.get('attempts')}")
            
    except Exception as e:
        print(f"Error in multi-transcriber test: {e}")

def main():
    """Run all tests."""
    print("Testing YouTube Transcription System")
    print("=" * 50)
    print()
    
    test_youtube_url_extraction()
    print()
    
    test_video_id_extraction()
    print()
    
    test_transcript_availability()
    print()
    
    test_multi_transcriber()
    print()
    
    print("=" * 50)
    print("Tests completed!")

if __name__ == "__main__":
    main()