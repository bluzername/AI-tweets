#!/usr/bin/env python3
"""
Test script to demonstrate posting tweets with images to Twitter/X.com
Shows the complete workflow: download thumbnail, create thread, post to Twitter.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from src.media_manager import MediaManager
from src.youtube_transcriber import YouTubeTranscriber
from src.thread_generator import Tweet
from src.x_publisher import XPublisher
from rich.console import Console
from rich.panel import Panel

console = Console()
load_dotenv()

def main():
    console.print(Panel.fit(
        "[bold cyan]Live Twitter Image Posting Test[/bold cyan]\n"
        "Complete end-to-end demonstration",
        border_style="cyan"
    ))

    # Step 1: Initialize components
    console.print("\n[bold]Step 1: Initialize Components[/bold]")
    media_manager = MediaManager(cache_dir="cache/images")
    yt_transcriber = YouTubeTranscriber()

    # Get Twitter credentials
    api_key = os.getenv('CASUAL_API_KEY')
    api_secret = os.getenv('CASUAL_API_SECRET')
    access_token = os.getenv('CASUAL_ACCESS_TOKEN')
    access_token_secret = os.getenv('CASUAL_ACCESS_TOKEN_SECRET')
    bearer_token = os.getenv('CASUAL_BEARER_TOKEN')

    if not all([api_key, api_secret, access_token, access_token_secret]):
        console.print("[red]✗ Missing Twitter API credentials in .env[/red]")
        return

    console.print("  ✓ MediaManager initialized")
    console.print("  ✓ YouTubeTranscriber initialized")
    console.print("  ✓ Twitter credentials loaded")

    # Step 2: Download YouTube thumbnail
    console.print("\n[bold]Step 2: Download YouTube Thumbnail[/bold]")
    youtube_url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
    video_id = yt_transcriber.extract_video_id(youtube_url)
    console.print(f"  Video ID: [green]{video_id}[/green]")

    thumbnail_url = yt_transcriber.get_thumbnail_url(video_id)
    console.print(f"  Thumbnail URL: [dim]{thumbnail_url}[/dim]")

    thumbnail_path = media_manager.download_and_cache_image(
        url=thumbnail_url,
        identifier=video_id
    )

    if not thumbnail_path:
        console.print("[red]  ✗ Failed to download thumbnail[/red]")
        return

    file_path = Path(thumbnail_path)
    file_size_kb = file_path.stat().st_size / 1024
    console.print(f"  ✓ Downloaded: [green]{thumbnail_path}[/green]")
    console.print(f"  ✓ File size: [green]{file_size_kb:.1f} KB[/green]")

    # Step 3: Create thread with image on first tweet
    console.print("\n[bold]Step 3: Create Thread with Image[/bold]")

    thread = [
        Tweet(
            content="🎙️ Mind-blowing insights from today's podcast episode!\n\n(Educational thread 🧵👇)",
            position=1,
            has_media=True,
            media_url=thumbnail_path
        ),
        Tweet(
            content="Key insight #1:\n\nSuccess isn't about working harder—it's about building better systems that compound over time.",
            position=2,
            has_media=False,
            media_url=None
        ),
        Tweet(
            content="Key insight #2:\n\nThe most valuable skill isn't knowledge—it's the ability to learn and adapt faster than others.",
            position=3,
            has_media=False,
            media_url=None
        ),
        Tweet(
            content="Key insight #3:\n\nFocus creates leverage. Saying 'no' to good opportunities makes room for great ones.",
            position=4,
            has_media=False,
            media_url=None
        ),
        Tweet(
            content="These insights changed how I think about productivity and growth.\n\nWhat's your biggest takeaway? 💬",
            position=5,
            has_media=False,
            media_url=None
        )
    ]

    console.print(f"  ✓ Created thread with {len(thread)} tweets")
    console.print(f"  ✓ First tweet has image: [green]{thread[0].has_media}[/green]")
    console.print(f"  ✓ Image path: [green]{thread[0].media_url}[/green]")

    # Step 4: Post to Twitter
    console.print("\n[bold]Step 4: Post Thread to Twitter/X.com[/bold]")

    publisher = XPublisher(
        api_key=api_key,
        api_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        bearer_token=bearer_token
    )

    console.print("  ✓ XPublisher initialized")
    console.print(f"  → Posting thread to Twitter...\n")

    # Post the thread
    try:
        result = publisher.publish_thread(
            thread=thread,
            podcast_name="Test Podcast",
            episode_title="Image Posting Test",
            dry_run=False
        )

        console.print(f"\n[bold]Result:[/bold] {result}")

        if result.get('success'):
            console.print(Panel.fit(
                f"[bold green]✅ Thread Posted Successfully![/bold green]\n\n"
                f"• Thread ID: {result.get('thread_id')}\n"
                f"• Tweets posted: {result.get('tweets_posted', 0)}\n"
                f"• First tweet with image: YES\n"
                f"• View on Twitter: https://twitter.com/user/status/{result.get('thread_id')}",
                border_style="green",
                title="Success"
            ))

            # Show tweet IDs
            console.print("\n[bold cyan]Tweet IDs:[/bold cyan]")
            tweet_ids = result.get('tweet_ids', [])
            for i, tweet_id in enumerate(tweet_ids, 1):
                media_indicator = "📷 [with image]" if i == 1 else ""
                console.print(f"  {i}. {tweet_id} {media_indicator}")
                console.print(f"     https://twitter.com/user/status/{tweet_id}")
        else:
            console.print(f"[red]✗ Failed to post thread: {result.get('error')}[/red]")
            console.print(f"[red]Full result: {result}[/red]")

    except Exception as e:
        console.print(f"[red]✗ Error posting thread: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

    console.print("\n" + "=" * 60)
    console.print("[bold green]Test completed![/bold green]")
    console.print("\nCheck your Twitter/X.com account to see the thread with the image!")


if __name__ == "__main__":
    main()
