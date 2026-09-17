#!/usr/bin/env python3
"""
Test script to verify image feature works end-to-end.
"""

import sys
from pathlib import Path

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

def show_header():
    """Display test header."""
    header = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🧪  IMAGE FEATURE TEST SUITE                          ║
║                                                          ║
║   Testing YouTube thumbnail extraction and upload       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """
    console.print(Panel(header, border_style="cyan", box=box.DOUBLE))
    console.print()


def test_1_media_manager():
    """Test 1: MediaManager can download and cache images."""
    console.print("\n[bold cyan]TEST 1: Media Manager[/bold cyan]")
    console.print("─" * 60)

    try:
        from src.media_manager import MediaManager

        manager = MediaManager(cache_dir="cache/images_test")

        # Test YouTube thumbnail URL construction
        video_id = "dQw4w9WgXcQ"  # Sample video ID
        thumbnail_url = manager.get_youtube_thumbnail_url(video_id)

        console.print(f"[dim]Video ID: {video_id}[/dim]")
        console.print(f"[dim]Thumbnail URL: {thumbnail_url}[/dim]")

        assert "youtube.com" in thumbnail_url
        assert video_id in thumbnail_url

        console.print("[green]✓ Thumbnail URL construction works[/green]")

        # Test download and cache
        console.print("\n[dim]Downloading test thumbnail...[/dim]")
        cached_path = manager.download_and_cache_image(
            url=thumbnail_url,
            identifier=f"test_{video_id}"
        )

        if cached_path:
            file_path = Path(cached_path)
            assert file_path.exists()
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            console.print(f"[green]✓ Image downloaded and cached ({file_size_mb:.2f}MB)[/green]")

            # Verify it's under 5MB
            assert file_size_mb <= 5.0
            console.print(f"[green]✓ Image size is within Twitter limits[/green]")

            # Test cache retrieval
            cached_again = manager.get_cached_image(f"test_{video_id}")
            assert cached_again == cached_path
            console.print(f"[green]✓ Cache retrieval works[/green]")

            # Get cache stats
            stats = manager.get_cache_stats()
            console.print(f"[dim]Cache stats: {stats['total_images']} images, {stats['total_size_mb']:.2f}MB[/dim]")
        else:
            console.print("[red]✗ Download failed[/red]")
            return False

        console.print("\n[bold green]✅ TEST 1 PASSED[/bold green]")
        return True

    except Exception as e:
        console.print(f"\n[bold red]❌ TEST 1 FAILED: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_2_youtube_transcriber():
    """Test 2: YouTube video ID extraction and thumbnail URL."""
    console.print("\n[bold cyan]TEST 2: YouTube Transcriber[/bold cyan]")
    console.print("─" * 60)

    try:
        from src.youtube_transcriber import YouTubeTranscriber

        yt = YouTubeTranscriber()

        # Test various URL formats
        test_urls = [
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]

        for url, expected_id in test_urls:
            video_id = yt.extract_video_id(url)
            console.print(f"[dim]URL: {url[:50]}...[/dim]")
            console.print(f"[dim]Extracted ID: {video_id}[/dim]")
            assert video_id == expected_id

        console.print(f"[green]✓ Video ID extraction works for {len(test_urls)} URL formats[/green]")

        # Test thumbnail URL
        thumbnail_url = yt.get_thumbnail_url("dQw4w9WgXcQ")
        assert "img.youtube.com" in thumbnail_url
        assert "dQw4w9WgXcQ" in thumbnail_url
        console.print(f"[green]✓ Thumbnail URL generation works[/green]")

        console.print("\n[bold green]✅ TEST 2 PASSED[/bold green]")
        return True

    except Exception as e:
        console.print(f"\n[bold red]❌ TEST 2 FAILED: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_3_database_schema():
    """Test 3: Database has thumbnail fields."""
    console.print("\n[bold cyan]TEST 3: Database Schema[/bold cyan]")
    console.print("─" * 60)

    try:
        import sqlite3

        # Initialize database (will run migrations)
        from src.podcast_ingestor import EpisodeDatabase
        db = EpisodeDatabase()

        # Check if thumbnail columns exist
        conn = sqlite3.connect('data/episodes.db')
        cursor = conn.execute("PRAGMA table_info(episodes)")
        columns = [row[1] for row in cursor.fetchall()]

        console.print(f"[dim]Database columns: {len(columns)} total[/dim]")

        assert "thumbnail_url" in columns
        console.print("[green]✓ thumbnail_url column exists[/green]")

        assert "thumbnail_local_path" in columns
        console.print("[green]✓ thumbnail_local_path column exists[/green]")

        conn.close()

        console.print("\n[bold green]✅ TEST 3 PASSED[/bold green]")
        return True

    except Exception as e:
        console.print(f"\n[bold red]❌ TEST 3 FAILED: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_4_tweet_metadata():
    """Test 4: Tweet can hold media metadata."""
    console.print("\n[bold cyan]TEST 4: Tweet Metadata[/bold cyan]")
    console.print("─" * 60)

    try:
        from src.thread_generator import Tweet

        # Create tweet with media
        tweet = Tweet(
            content="Test tweet with image",
            position=1,
            has_media=True,
            media_url="cache/images/test.jpg"
        )

        assert tweet.has_media == True
        console.print("[green]✓ has_media field works[/green]")

        assert tweet.media_url == "cache/images/test.jpg"
        console.print("[green]✓ media_url field works[/green]")

        assert tweet.is_valid()
        console.print("[green]✓ Tweet validation works[/green]")

        console.print("\n[bold green]✅ TEST 4 PASSED[/bold green]")
        return True

    except Exception as e:
        console.print(f"\n[bold red]❌ TEST 4 FAILED: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_5_x_publisher_media_upload():
    """Test 5: XPublisher has media upload method."""
    console.print("\n[bold cyan]TEST 5: X Publisher Media Upload[/bold cyan]")
    console.print("─" * 60)

    try:
        from src.x_publisher import XPublisher

        # Initialize without credentials (will use fallback)
        publisher = XPublisher()

        # Check that _upload_media method exists
        assert hasattr(publisher, '_upload_media')
        console.print("[green]✓ _upload_media method exists[/green]")

        # Check that API attribute exists (even if None)
        assert hasattr(publisher, 'api')
        console.print("[green]✓ API attribute exists for v1.1[/green]")

        console.print("[yellow]ℹ️  Cannot test actual Twitter upload without credentials[/yellow]")

        console.print("\n[bold green]✅ TEST 5 PASSED[/bold green]")
        return True

    except Exception as e:
        console.print(f"\n[bold red]❌ TEST 5 FAILED: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_6_integration():
    """Test 6: End-to-end integration with mock episode."""
    console.print("\n[bold cyan]TEST 6: Integration Test[/bold cyan]")
    console.print("─" * 60)

    try:
        from src.podcast_ingestor import PodcastEpisode
        from src.youtube_transcriber import YouTubeTranscriber
        from src.media_manager import MediaManager
        from datetime import datetime

        # Create mock episode with YouTube URL
        episode = PodcastEpisode(
            title="Test Episode",
            podcast_name="Test Podcast",
            audio_url="https://example.com/audio.mp3",
            published_date=datetime.now(),
            description="Test description",
            youtube_urls=["https://youtube.com/watch?v=dQw4w9WgXcQ"]
        )

        console.print(f"[dim]Created mock episode with YouTube URL[/dim]")

        # Extract video ID
        yt = YouTubeTranscriber()
        video_id = yt.extract_video_id(episode.youtube_urls[0])
        console.print(f"[dim]Extracted video ID: {video_id}[/dim]")
        assert video_id is not None

        # Get thumbnail URL
        thumbnail_url = yt.get_thumbnail_url(video_id)
        console.print(f"[dim]Thumbnail URL: {thumbnail_url[:60]}...[/dim]")

        # Download thumbnail
        manager = MediaManager(cache_dir="cache/images_test")
        thumbnail_path = manager.download_and_cache_image(
            url=thumbnail_url,
            identifier=f"integration_test_{video_id}"
        )

        assert thumbnail_path is not None
        assert Path(thumbnail_path).exists()
        console.print(f"[green]✓ Complete flow works: YouTube URL → video ID → thumbnail → cache[/green]")

        console.print("\n[bold green]✅ TEST 6 PASSED[/bold green]")
        return True

    except Exception as e:
        console.print(f"\n[bold red]❌ TEST 6 FAILED: {str(e)}[/bold red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def main():
    """Run all tests."""
    show_header()

    tests = [
        test_1_media_manager,
        test_2_youtube_transcriber,
        test_3_database_schema,
        test_4_tweet_metadata,
        test_5_x_publisher_media_upload,
        test_6_integration
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    # Summary
    console.print("\n" + "═" * 60)

    summary_table = Table(title="Test Summary", box=box.DOUBLE_EDGE, show_header=True, header_style="bold magenta")
    summary_table.add_column("Test", style="cyan", width=40)
    summary_table.add_column("Result", justify="center", width=15)

    test_names = [
        "Media Manager",
        "YouTube Transcriber",
        "Database Schema",
        "Tweet Metadata",
        "X Publisher",
        "Integration"
    ]

    for name, passed in zip(test_names, results):
        status = "[green]✅ PASSED[/green]" if passed else "[red]❌ FAILED[/red]"
        summary_table.add_row(name, status)

    console.print(summary_table)

    passed_count = sum(results)
    total_count = len(results)

    console.print()
    if passed_count == total_count:
        console.print(f"[bold green]🎉 ALL TESTS PASSED ({passed_count}/{total_count})[/bold green]")
        console.print("[bold green]Image feature is working flawlessly! ✨[/bold green]")
        return 0
    else:
        console.print(f"[bold red]❌ SOME TESTS FAILED ({passed_count}/{total_count})[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
