"""
Media Manager - Handles downloading, caching, and optimizing images for tweets.
"""

import logging
import os
import requests
from pathlib import Path
from typing import Optional, Dict
from PIL import Image
import hashlib

# Rich library for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

logger = logging.getLogger(__name__)
console = Console()


class MediaManager:
    """
    Manages media files (images) for tweets.

    Features:
    - Downloads images from URLs
    - Caches to avoid re-downloading
    - Resizes/optimizes for Twitter (max 5MB)
    - Supports YouTube thumbnails
    """

    def __init__(self, cache_dir: str = "cache/images"):
        """
        Initialize MediaManager with cache directory.

        Args:
            cache_dir: Directory to store cached images
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Twitter image constraints
        self.max_file_size = 5 * 1024 * 1024  # 5MB
        self.max_dimensions = (4096, 4096)  # Twitter max
        self.supported_formats = ['JPEG', 'PNG', 'GIF', 'WEBP']

        logger.info(f"MediaManager initialized with cache: {self.cache_dir}")

    def get_youtube_thumbnail_url(self, video_id: str, quality: str = "maxresdefault") -> str:
        """
        Get YouTube thumbnail URL for a video.

        Args:
            video_id: YouTube video ID
            quality: Thumbnail quality (maxresdefault, hqdefault, mqdefault, default)
                    - maxresdefault: 1280x720 (best, may not exist for all videos)
                    - hqdefault: 480x360 (high quality, always exists)
                    - mqdefault: 320x180 (medium quality)
                    - default: 120x90 (low quality)

        Returns:
            URL to YouTube thumbnail
        """
        return f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"

    def download_and_cache_image(
        self,
        url: str,
        identifier: str,
        force_refresh: bool = False
    ) -> Optional[str]:
        """
        Download and cache an image from URL.

        Args:
            url: URL to download image from
            identifier: Unique identifier for caching (e.g., video_id, episode_id)
            force_refresh: Force re-download even if cached

        Returns:
            Path to cached image file, or None if failed
        """
        try:
            # Generate cache filename from identifier
            cache_filename = f"{identifier}.jpg"
            cache_path = self.cache_dir / cache_filename

            # Check if already cached
            if cache_path.exists() and not force_refresh:
                console.print(f"[dim]  ✓ Image already cached: {cache_filename}[/dim]")
                return str(cache_path)

            console.print(f"[dim]  Downloading image from URL...[/dim]")

            # Download image
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("[yellow]Downloading image...", total=None)

                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                # Get content length if available
                total_size = int(response.headers.get('content-length', 0))
                if total_size:
                    progress.update(task, total=total_size)

                # Download to temporary file
                temp_path = cache_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                progress.update(task, completed=downloaded)

            # Optimize and move to final location
            optimized_path = self._optimize_image(temp_path, cache_path)

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

            if optimized_path:
                file_size = optimized_path.stat().st_size
                size_mb = file_size / (1024 * 1024)
                console.print(f"[dim]  ✓ Image cached ({size_mb:.2f}MB): {cache_filename}[/dim]")
                return str(optimized_path)
            else:
                console.print(f"[red]  ✗ Failed to optimize image[/red]")
                return None

        except Exception as e:
            console.print(f"[red]  ✗ Image download failed: {str(e)[:60]}...[/red]")
            logger.error(f"Failed to download image from {url}: {e}")
            return None

    def _optimize_image(self, input_path: Path, output_path: Path) -> Optional[Path]:
        """
        Optimize image for Twitter (resize, compress, convert format if needed).

        Args:
            input_path: Path to input image
            output_path: Path to save optimized image

        Returns:
            Path to optimized image, or None if failed
        """
        try:
            # Open image
            with Image.open(input_path) as img:
                # Convert RGBA to RGB if necessary (Twitter doesn't like RGBA JPEGs)
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large
                if img.size[0] > self.max_dimensions[0] or img.size[1] > self.max_dimensions[1]:
                    img.thumbnail(self.max_dimensions, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {img.size}")

                # Save with compression
                quality = 95
                img.save(output_path, 'JPEG', quality=quality, optimize=True)

                # Check file size and reduce quality if needed
                while output_path.stat().st_size > self.max_file_size and quality > 50:
                    quality -= 10
                    img.save(output_path, 'JPEG', quality=quality, optimize=True)
                    logger.info(f"Reduced quality to {quality} to meet size limit")

                # Verify final size
                final_size = output_path.stat().st_size
                if final_size > self.max_file_size:
                    logger.warning(f"Image still too large ({final_size} bytes), may fail on Twitter")

                return output_path

        except Exception as e:
            logger.error(f"Failed to optimize image: {e}")
            return None

    def get_cached_image(self, identifier: str) -> Optional[str]:
        """
        Get path to cached image if it exists.

        Args:
            identifier: Unique identifier for the image

        Returns:
            Path to cached image, or None if not cached
        """
        cache_path = self.cache_dir / f"{identifier}.jpg"
        if cache_path.exists():
            return str(cache_path)
        return None

    def clear_cache(self, older_than_days: int = 30):
        """
        Clear old cached images.

        Args:
            older_than_days: Remove images older than this many days
        """
        import time
        current_time = time.time()
        cutoff_time = current_time - (older_than_days * 24 * 60 * 60)

        removed_count = 0
        for image_file in self.cache_dir.glob("*.jpg"):
            if image_file.stat().st_mtime < cutoff_time:
                image_file.unlink()
                removed_count += 1

        if removed_count > 0:
            console.print(f"[dim]  Removed {removed_count} old cached images[/dim]")
            logger.info(f"Cleared {removed_count} images from cache")

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the image cache.

        Returns:
            Dictionary with cache statistics
        """
        image_files = list(self.cache_dir.glob("*.jpg"))
        total_size = sum(f.stat().st_size for f in image_files)

        return {
            "total_images": len(image_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
