"""
Debunk Thread Generator - Creates fact-check threads for the PodDebunker account.

This module formats verified false/misleading claims into engaging Twitter threads
with a consistent "debunker" personality that is objective and data-driven.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

from .fact_checker import FactCheckResult, Verdict
from .tweet_splitter import TweetSplitter, ThreadAssembler
from .thread_generator import Tweet, TweetFormat

logger = logging.getLogger(__name__)


@dataclass
class DebunkThreadConfig:
    """Configuration for debunk thread generation."""
    min_false_claims: int = 2             # Minimum false claims to publish
    include_misleading: bool = True       # Include MISLEADING verdicts (not just FALSE)
    add_thumbnail: bool = True            # Add branded thumbnail to hook tweet
    thumbnail_cache_dir: str = "cache/thumbnails"


class DebunkThumbnailGenerator:
    """
    Generates branded thumbnails for fact-check threads.

    Creates an image with:
    - Podcast artwork as background
    - "FACT CHECK" badge overlay
    - PodDebunker branding
    """

    def __init__(self, cache_dir: str = "cache/thumbnails"):
        """
        Initialize thumbnail generator.

        Args:
            cache_dir: Directory to cache generated thumbnails
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Badge colors
        self.badge_color_false = (220, 53, 69)      # Red for false claims
        self.badge_color_misleading = (255, 193, 7)  # Yellow/amber for misleading
        self.text_color = (255, 255, 255)           # White text
        self.overlay_color = (0, 0, 0, 180)         # Semi-transparent black

    def create_thumbnail(
        self,
        podcast_artwork_url: Optional[str],
        podcast_name: str,
        episode_title: str,
        false_claim_count: int,
        has_misleading: bool = False
    ) -> Optional[str]:
        """
        Create a branded fact-check thumbnail.

        Args:
            podcast_artwork_url: URL to podcast artwork (or None for default)
            podcast_name: Name of the podcast
            episode_title: Title of the episode
            false_claim_count: Number of false claims found
            has_misleading: Whether there are misleading (vs pure false) claims

        Returns:
            Path to generated thumbnail, or None if failed
        """
        try:
            # Generate cache key
            cache_key = f"factcheck_{podcast_name}_{episode_title}".replace(" ", "_")[:50]
            cache_path = self.cache_dir / f"{cache_key}.jpg"

            # Check cache
            if cache_path.exists():
                logger.debug(f"Using cached thumbnail: {cache_path}")
                return str(cache_path)

            # Get base image
            if podcast_artwork_url:
                base_image = self._download_image(podcast_artwork_url)
            else:
                base_image = self._create_default_background()

            if not base_image:
                base_image = self._create_default_background()

            # Resize to standard dimensions (1200x675 for Twitter card)
            base_image = self._resize_image(base_image, (1200, 675))

            # Add overlays
            final_image = self._add_overlays(
                base_image,
                podcast_name,
                false_claim_count,
                has_misleading
            )

            # Save
            final_image.save(cache_path, "JPEG", quality=90)
            logger.info(f"Generated thumbnail: {cache_path}")

            return str(cache_path)

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return None

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.warning(f"Failed to download image: {e}")
            return None

    def _create_default_background(self) -> Image.Image:
        """Create a default background image."""
        # Create a gradient background
        width, height = 1200, 675
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)

        # Dark gradient from top to bottom
        for y in range(height):
            # Gradient from dark blue to darker blue
            r = int(20 + (y / height) * 15)
            g = int(30 + (y / height) * 20)
            b = int(50 + (y / height) * 30)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        return image

    def _resize_image(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """Resize and crop image to target size, maintaining aspect ratio."""
        target_w, target_h = target_size
        img_w, img_h = image.size

        # Calculate scaling to cover target area
        scale = max(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Crop to center
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h

        return image.crop((left, top, right, bottom))

    def _add_overlays(
        self,
        image: Image.Image,
        podcast_name: str,
        false_claim_count: int,
        has_misleading: bool
    ) -> Image.Image:
        """Add text overlays and badges to the image."""
        # Convert to RGBA for transparency support
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        # Create overlay layer
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Add semi-transparent dark overlay at bottom
        overlay_height = 200
        draw.rectangle(
            [(0, image.height - overlay_height), (image.width, image.height)],
            fill=(0, 0, 0, 180)
        )

        # Add "FACT CHECK" badge at top
        badge_color = self.badge_color_misleading if has_misleading else self.badge_color_false
        badge_text = "FACT CHECK"
        badge_padding = 20
        badge_height = 60

        # Draw badge background
        draw.rectangle(
            [(0, 0), (image.width, badge_height)],
            fill=badge_color + (230,)  # Add alpha
        )

        # Try to load a font, fall back to default
        try:
            # Try common font paths
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "C:\\Windows\\Fonts\\arial.ttf"
            ]
            font_large = None
            font_medium = None

            for font_path in font_paths:
                if os.path.exists(font_path):
                    font_large = ImageFont.truetype(font_path, 36)
                    font_medium = ImageFont.truetype(font_path, 28)
                    break

            if not font_large:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()

        except Exception:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()

        # Draw badge text
        badge_bbox = draw.textbbox((0, 0), badge_text, font=font_large)
        badge_text_width = badge_bbox[2] - badge_bbox[0]
        draw.text(
            ((image.width - badge_text_width) // 2, 12),
            badge_text,
            font=font_large,
            fill=self.text_color
        )

        # Draw claim count
        claim_text = f"{false_claim_count} claim{'s' if false_claim_count != 1 else ''} verified"
        draw.text(
            (40, image.height - overlay_height + 40),
            claim_text,
            font=font_large,
            fill=self.text_color
        )

        # Draw podcast name
        podcast_display = podcast_name[:40] + "..." if len(podcast_name) > 40 else podcast_name
        draw.text(
            (40, image.height - overlay_height + 100),
            podcast_display,
            font=font_medium,
            fill=(200, 200, 200)
        )

        # Draw PodDebunker branding
        brand_text = "@PodDebunker"
        brand_bbox = draw.textbbox((0, 0), brand_text, font=font_medium)
        brand_width = brand_bbox[2] - brand_bbox[0]
        draw.text(
            (image.width - brand_width - 40, image.height - 50),
            brand_text,
            font=font_medium,
            fill=(150, 150, 150)
        )

        # Composite overlay onto image
        image = Image.alpha_composite(image, overlay)

        # Convert back to RGB for saving as JPEG
        return image.convert('RGB')


class DebunkThreadGenerator:
    """
    Generates Twitter threads for fact-checked podcast claims.

    Creates threads with:
    - Hook tweet with thumbnail
    - Body tweets for each false/misleading claim
    - Closer tweet with CTA
    """

    def __init__(self, config: Optional[DebunkThreadConfig] = None):
        """
        Initialize debunk thread generator.

        Args:
            config: Configuration options
        """
        self.config = config or DebunkThreadConfig()
        self.splitter = TweetSplitter()
        self.assembler = ThreadAssembler(self.splitter)
        self.thumbnail_generator = DebunkThumbnailGenerator(
            self.config.thumbnail_cache_dir
        )

        logger.info(f"DebunkThreadGenerator initialized (min_claims: {self.config.min_false_claims})")

    def generate_thread(
        self,
        podcast_name: str,
        episode_title: str,
        false_claims: List[FactCheckResult],
        podcast_artwork_url: Optional[str] = None
    ) -> List[Tweet]:
        """
        Generate a fact-check thread from verified false claims.

        Args:
            podcast_name: Name of the podcast
            episode_title: Title of the episode
            false_claims: List of FactCheckResult with FALSE/MISLEADING verdicts
            podcast_artwork_url: URL to podcast artwork for thumbnail

        Returns:
            List of Tweet objects, or empty list if insufficient claims
        """
        # Filter to only false/misleading claims
        relevant_claims = [
            c for c in false_claims
            if c.verdict == Verdict.FALSE or
               (self.config.include_misleading and c.verdict == Verdict.MISLEADING)
        ]

        # Check minimum threshold
        if len(relevant_claims) < self.config.min_false_claims:
            logger.info(
                f"Insufficient false claims ({len(relevant_claims)}/{self.config.min_false_claims}), "
                "skipping thread generation"
            )
            return []

        logger.info(f"Generating debunk thread for {len(relevant_claims)} claims")

        # Generate thumbnail
        thumbnail_path = None
        if self.config.add_thumbnail:
            has_misleading = any(c.verdict == Verdict.MISLEADING for c in relevant_claims)
            thumbnail_path = self.thumbnail_generator.create_thumbnail(
                podcast_artwork_url=podcast_artwork_url,
                podcast_name=podcast_name,
                episode_title=episode_title,
                false_claim_count=len(relevant_claims),
                has_misleading=has_misleading
            )

        # Generate hook tweet
        hook = self._generate_hook(podcast_name, episode_title, len(relevant_claims))

        # Generate body content for each claim
        body_tweets = []
        for i, claim_result in enumerate(relevant_claims, 1):
            claim_tweets = self._format_claim_tweets(claim_result, i, len(relevant_claims))
            body_tweets.extend(claim_tweets)

        # Generate closer tweet
        closer = self._generate_closer()

        # Assemble into Tweet objects
        tweets = []
        position = 1

        # Hook with thumbnail
        tweets.append(Tweet(
            content=hook,
            position=position,
            has_media=thumbnail_path is not None,
            media_url=thumbnail_path,
            tweet_format=TweetFormat.THREAD
        ))
        position += 1

        # Body tweets
        for tweet_content in body_tweets:
            tweets.append(Tweet(
                content=tweet_content,
                position=position,
                has_media=False,
                tweet_format=TweetFormat.THREAD
            ))
            position += 1

        # Closer
        tweets.append(Tweet(
            content=closer,
            position=position,
            has_media=False,
            tweet_format=TweetFormat.THREAD
        ))

        logger.info(f"Generated thread with {len(tweets)} tweets")
        return tweets

    def _generate_hook(self, podcast_name: str, episode_title: str, claim_count: int) -> str:
        """Generate the opening hook tweet."""
        # Truncate episode title if needed to fit in tweet
        max_title_len = 80
        title_display = episode_title[:max_title_len] + "..." if len(episode_title) > max_title_len else episode_title

        hook_templates = [
            f'🔍 I fact-checked {podcast_name}\'s latest episode "{title_display}"\n\nFound {claim_count} claim{"s" if claim_count != 1 else ""} that needed a reality check. 🧵👇',
            f'🔍 Listened to {podcast_name} so you don\'t have to check the facts.\n\n"{title_display}"\n\n{claim_count} claim{"s" if claim_count != 1 else ""} verified below 🧵👇',
            f'🔎 FACT CHECK: {podcast_name}\n\n"{title_display}"\n\n{claim_count} claim{"s" if claim_count != 1 else ""} that didn\'t hold up to scrutiny 🧵👇'
        ]

        # Use first template that fits
        for template in hook_templates:
            if len(template) <= 280:
                return template

        # Fallback with shorter title
        return f'🔍 FACT CHECK: {podcast_name}\n\n{claim_count} claims verified 🧵👇'

    def _format_claim_tweets(
        self,
        claim_result: FactCheckResult,
        claim_number: int,
        total_claims: int
    ) -> List[str]:
        """
        Format a single fact-check result into tweets.

        Args:
            claim_result: The fact-check result
            claim_number: Position in the list (1-indexed)
            total_claims: Total number of claims being fact-checked

        Returns:
            List of tweet strings for this claim
        """
        # Get the claim text (truncate if very long)
        claim_text = claim_result.original_claim.claim_text
        if len(claim_text) > 200:
            claim_text = claim_text[:197] + "..."

        # Get correction
        correction = claim_result.correction or "This claim could not be verified with available evidence."

        # Use the splitter for proper formatting
        tweets = self.splitter.split_fact_check_content(
            claim_text=claim_text,
            correction=correction,
            source_name=claim_result.source_name,
            source_url=claim_result.source_url
        )

        # Add claim number prefix to first tweet if multiple claims
        if total_claims > 1 and tweets:
            prefix = f"[{claim_number}/{total_claims}] "
            if len(prefix + tweets[0]) <= 280:
                tweets[0] = prefix + tweets[0]

        return tweets

    def _generate_closer(self) -> str:
        """Generate the closing tweet with CTA."""
        closers = [
            "💡 Misinformation spreads fast. Always verify claims before sharing.\n\nFollow @PodDebunker for weekly podcast fact-checks.\n\nGot an episode you want checked? Drop it below 👇",
            "📌 That's the breakdown. Remember: extraordinary claims need extraordinary evidence.\n\nFollow @PodDebunker for more fact-checks.\n\nWhat podcast should I check next? 👇",
            "✅ Always check your sources.\n\nFollow @PodDebunker for regular podcast fact-checks.\n\nKnow a podcast that needs fact-checking? Let me know 👇"
        ]

        # Return first one that fits
        for closer in closers:
            if len(closer) <= 280:
                return closer

        return closers[0][:280]

    def should_publish(self, false_claims: List[FactCheckResult]) -> bool:
        """
        Check if there are enough false claims to warrant publishing.

        Args:
            false_claims: List of fact-check results

        Returns:
            True if thread should be published
        """
        relevant_count = sum(
            1 for c in false_claims
            if c.verdict == Verdict.FALSE or
               (self.config.include_misleading and c.verdict == Verdict.MISLEADING)
        )
        return relevant_count >= self.config.min_false_claims
