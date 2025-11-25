#!/usr/bin/env python3
"""
Demonstration of the image feature working end-to-end.
Shows how thumbnails are downloaded, cached, and would be posted to Twitter.
"""

from src.media_manager import MediaManager
from src.youtube_transcriber import YouTubeTranscriber
from src.thread_generator import Tweet
from rich.console import Console
from rich.panel import Panel
from pathlib import Path

console = Console()

def main():
    console.print(Panel.fit(
        "[bold cyan]Image Feature Demonstration[/bold cyan]\n"
        "Shows complete thumbnail workflow",
        border_style="cyan"
    ))

    # Step 1: Initialize components
    console.print("\n[bold]Step 1: Initialize Components[/bold]")
    media_manager = MediaManager(cache_dir="cache/images")
    yt_transcriber = YouTubeTranscriber()
    console.print("  ✓ MediaManager initialized")
    console.print("  ✓ YouTubeTranscriber initialized")

    # Step 2: Process YouTube URL
    console.print("\n[bold]Step 2: Extract Video ID from URL[/bold]")
    youtube_url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
    console.print(f"  URL: {youtube_url}")

    video_id = yt_transcriber.extract_video_id(youtube_url)
    console.print(f"  ✓ Extracted video ID: [green]{video_id}[/green]")

    # Step 3: Get thumbnail URL
    console.print("\n[bold]Step 3: Get Thumbnail URL[/bold]")
    thumbnail_url = yt_transcriber.get_thumbnail_url(video_id)
    console.print(f"  ✓ Thumbnail URL: [dim]{thumbnail_url}[/dim]")

    # Step 4: Download and cache thumbnail
    console.print("\n[bold]Step 4: Download & Cache Thumbnail[/bold]")
    thumbnail_path = media_manager.download_and_cache_image(
        url=thumbnail_url,
        identifier=video_id
    )

    if thumbnail_path:
        file_path = Path(thumbnail_path)
        file_size_kb = file_path.stat().st_size / 1024
        console.print(f"  ✓ Downloaded: [green]{thumbnail_path}[/green]")
        console.print(f"  ✓ File size: [green]{file_size_kb:.1f} KB[/green]")
        console.print(f"  ✓ Exists: [green]{file_path.exists()}[/green]")
    else:
        console.print("  [red]✗ Download failed[/red]")
        return

    # Step 5: Create tweet with image
    console.print("\n[bold]Step 5: Create Tweet with Media[/bold]")
    tweet = Tweet(
        content="🎙️ New podcast episode:\n\nMind-blowing insights on building billion-dollar companies.\n\n(Thread 🧵)",
        position=1,
        has_media=True,
        media_url=thumbnail_path
    )

    console.print(f"  ✓ Tweet created with media attachment")
    console.print(f"  ✓ Content: [dim]{tweet.content[:50]}...[/dim]")
    console.print(f"  ✓ Has media: [green]{tweet.has_media}[/green]")
    console.print(f"  ✓ Media URL: [green]{tweet.media_url}[/green]")

    # Step 6: Show what would happen on Twitter
    console.print("\n[bold]Step 6: Twitter Posting Flow[/bold]")
    console.print("  [dim]If Twitter credentials were configured:[/dim]")
    console.print("    1. Upload media file via API v1.1 → [cyan]get media_id[/cyan]")
    console.print(f"       [dim]POST media/upload with file: {thumbnail_path}[/dim]")
    console.print("    2. Create tweet via API v2 with [cyan]media_ids[/cyan]")
    console.print(f"       [dim]POST tweets with text + media_ids=[media_id][/dim]")
    console.print("    3. [green]Tweet appears with thumbnail image![/green]")

    # Step 7: Show cache stats
    console.print("\n[bold]Step 7: Cache Statistics[/bold]")
    stats = media_manager.get_cache_stats()
    console.print(f"  ✓ Total images cached: [green]{stats['total_images']}[/green]")
    console.print(f"  ✓ Total cache size: [green]{stats['total_size_mb']:.2f} MB[/green]")
    console.print(f"  ✓ Cache directory: [green]{stats['cache_dir']}[/green]")

    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        "[bold green]✅ Image Feature Working Perfectly![/bold green]\n\n"
        "• Downloaded real YouTube thumbnail\n"
        "• Cached locally for reuse\n"
        "• Created tweet with media attachment\n"
        "• Ready for Twitter API posting",
        border_style="green",
        title="Success"
    ))

    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("  1. Add Twitter API credentials to .env")
    console.print("  2. Run the full pipeline")
    console.print("  3. Images will automatically attach to first tweet")
    console.print("\n[dim]Image file saved at: {thumbnail_path}[/dim]".format(thumbnail_path=thumbnail_path))


if __name__ == "__main__":
    main()
