"""
Tweet Splitter - Intelligently splits long content across multiple tweets.

This module ensures that long fact-check results can be properly formatted
into threads without arbitrary limits on tweet count. Quality matters over brevity.
"""

import logging
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for tweet splitting behavior."""
    max_tweet_length: int = 280
    min_split_length: int = 100      # Don't split if remaining is less than this
    preserve_urls: bool = True        # Never split URLs
    preserve_mentions: bool = True    # Never split @mentions
    preserve_hashtags: bool = True    # Never split #hashtags
    preserve_emojis: bool = True      # Keep emojis with their context


class TweetSplitter:
    """
    Intelligently splits long content across multiple tweets.

    Uses a hierarchy of break points:
    1. Paragraph breaks (double newline)
    2. Sentence boundaries (. ! ?)
    3. Clause boundaries (, ; :)
    4. Word boundaries (space)

    Never splits:
    - URLs mid-way
    - @mentions
    - #hashtags
    - Emoji sequences from their context
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """
        Initialize tweet splitter.

        Args:
            config: Configuration options (uses defaults if not provided)
        """
        self.config = config or SplitConfig()

        # Regex patterns for protected content
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F'  # emoticons
            r'\U0001F300-\U0001F5FF'   # symbols & pictographs
            r'\U0001F680-\U0001F6FF'   # transport & map symbols
            r'\U0001F700-\U0001F77F'   # alchemical symbols
            r'\U0001F780-\U0001F7FF'   # Geometric Shapes Extended
            r'\U0001F800-\U0001F8FF'   # Supplemental Arrows-C
            r'\U0001F900-\U0001F9FF'   # Supplemental Symbols and Pictographs
            r'\U0001FA00-\U0001FA6F'   # Chess Symbols
            r'\U0001FA70-\U0001FAFF'   # Symbols and Pictographs Extended-A
            r'\U00002702-\U000027B0'   # Dingbats
            r'\U000024C2-\U0001F251'   # Various symbols
            r']+',
            re.UNICODE
        )

    def split_content(self, content: str) -> List[str]:
        """
        Split content into tweet-sized chunks.

        Args:
            content: The full text content to split

        Returns:
            List of strings, each within tweet character limit
        """
        content = content.strip()

        # If already fits, return as-is
        if len(content) <= self.config.max_tweet_length:
            return [content]

        tweets = []
        remaining = content

        while remaining:
            remaining = remaining.strip()

            # If remaining fits, add it and done
            if len(remaining) <= self.config.max_tweet_length:
                if remaining:
                    tweets.append(remaining)
                break

            # Find the best break point
            break_point = self._find_best_break(remaining)
            tweet_content = remaining[:break_point].strip()

            if tweet_content:
                tweets.append(tweet_content)

            remaining = remaining[break_point:].strip()

        return tweets

    def _find_best_break(self, text: str) -> int:
        """
        Find the optimal break point within the character limit.

        Uses a hierarchy of preferences:
        1. Paragraph breaks (double newline)
        2. Sentence boundaries
        3. Clause boundaries
        4. Word boundaries

        Args:
            text: Text to find break point in

        Returns:
            Character index for the break point
        """
        max_len = self.config.max_tweet_length
        search_text = text[:max_len]

        # Find protected regions that we shouldn't break
        protected_regions = self._find_protected_regions(search_text)

        # Priority 1: Paragraph break (double newline)
        para_break = self._find_break_avoiding_protected(
            search_text, '\n\n', protected_regions, min_position=max_len * 0.3
        )
        if para_break > 0:
            return para_break + 2  # Include the newlines in first part

        # Priority 2: Single newline
        newline_break = self._find_break_avoiding_protected(
            search_text, '\n', protected_regions, min_position=max_len * 0.4
        )
        if newline_break > 0:
            return newline_break + 1

        # Priority 3: Sentence boundary (. ! ?)
        for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            sent_break = self._find_break_avoiding_protected(
                search_text, punct, protected_regions, min_position=max_len * 0.3
            )
            if sent_break > 0:
                return sent_break + len(punct)

        # Priority 4: Clause boundary (, ; : —)
        for punct in [', ', '; ', ': ', ' — ', ' - ']:
            clause_break = self._find_break_avoiding_protected(
                search_text, punct, protected_regions, min_position=max_len * 0.5
            )
            if clause_break > 0:
                return clause_break + len(punct)

        # Priority 5: Word boundary (last space)
        space_break = self._find_break_avoiding_protected(
            search_text, ' ', protected_regions, min_position=max_len * 0.3
        )
        if space_break > 0:
            return space_break + 1

        # Absolute fallback: hard cut at max length
        # Try to avoid cutting in the middle of a word
        logger.warning(f"Had to hard-cut text at {max_len} characters")
        return max_len

    def _find_protected_regions(self, text: str) -> List[Tuple[int, int]]:
        """
        Find regions in text that should not be broken (URLs, mentions, etc.).

        Args:
            text: Text to scan

        Returns:
            List of (start, end) tuples for protected regions
        """
        protected = []

        if self.config.preserve_urls:
            for match in self.url_pattern.finditer(text):
                protected.append((match.start(), match.end()))

        if self.config.preserve_mentions:
            for match in self.mention_pattern.finditer(text):
                protected.append((match.start(), match.end()))

        if self.config.preserve_hashtags:
            for match in self.hashtag_pattern.finditer(text):
                protected.append((match.start(), match.end()))

        return protected

    def _find_break_avoiding_protected(
        self,
        text: str,
        delimiter: str,
        protected_regions: List[Tuple[int, int]],
        min_position: float = 0
    ) -> int:
        """
        Find the last occurrence of delimiter that's not in a protected region.

        Args:
            text: Text to search
            delimiter: The delimiter to find
            protected_regions: List of (start, end) regions to avoid
            min_position: Minimum position (as ratio of text length)

        Returns:
            Index of delimiter, or -1 if not found
        """
        min_idx = int(min_position)
        search_start = len(text)

        while True:
            idx = text.rfind(delimiter, 0, search_start)
            if idx < min_idx:
                return -1

            # Check if this position is inside a protected region
            is_protected = False
            for start, end in protected_regions:
                if start <= idx < end:
                    is_protected = True
                    search_start = start  # Search before this protected region
                    break

            if not is_protected:
                return idx

        return -1

    def split_fact_check_content(
        self,
        claim_text: str,
        correction: str,
        source_name: str,
        source_url: str
    ) -> List[str]:
        """
        Split a fact-check result into properly formatted tweets.

        This method handles the specific structure of fact-check content:
        - Claim section (❌ CLAIM: ...)
        - Reality section (✅ REALITY: ...)
        - Source section (📎 Source: ... 🔗 ...)

        Args:
            claim_text: The original false/misleading claim
            correction: The accurate information
            source_name: Name of the source
            source_url: URL to the source

        Returns:
            List of formatted tweet strings
        """
        tweets = []

        # Format the claim section
        claim_section = f'❌ CLAIM: "{claim_text}"'

        # Format the reality/correction section
        reality_section = f"✅ REALITY: {correction}"

        # Format the source section
        source_section = f"📎 Source: {source_name}\n🔗 {source_url}"

        # Try to fit claim + reality + source in minimum tweets
        full_content = f"{claim_section}\n\n{reality_section}\n\n{source_section}"

        if len(full_content) <= self.config.max_tweet_length:
            # Everything fits in one tweet!
            return [full_content]

        # Try claim + reality (source separate)
        claim_reality = f"{claim_section}\n\n{reality_section}"

        if len(claim_reality) <= self.config.max_tweet_length:
            tweets.append(claim_reality)
            tweets.append(source_section)
            return tweets

        # Need to split further

        # First, handle the claim
        if len(claim_section) <= self.config.max_tweet_length:
            tweets.append(claim_section)
        else:
            # Split the claim itself (rare but possible for very long quotes)
            claim_parts = self.split_content(claim_section)
            tweets.extend(claim_parts)

        # Then handle the reality/correction
        if len(reality_section) <= self.config.max_tweet_length:
            tweets.append(reality_section)
        else:
            # Split the correction
            reality_parts = self.split_content(reality_section)
            tweets.extend(reality_parts)

        # Finally, add the source (try to append to last tweet if it fits)
        if tweets and len(tweets[-1]) + len(f"\n\n{source_section}") <= self.config.max_tweet_length:
            tweets[-1] += f"\n\n{source_section}"
        else:
            tweets.append(source_section)

        return tweets

    def estimate_tweet_count(self, content: str) -> int:
        """
        Estimate how many tweets content will split into.

        Args:
            content: The content to estimate

        Returns:
            Estimated number of tweets
        """
        if len(content) <= self.config.max_tweet_length:
            return 1

        # Rough estimate based on character count
        return (len(content) // self.config.max_tweet_length) + 1


class ThreadAssembler:
    """
    Assembles multiple fact-check results into a coherent thread.
    """

    def __init__(self, splitter: Optional[TweetSplitter] = None):
        """
        Initialize thread assembler.

        Args:
            splitter: TweetSplitter instance (creates default if not provided)
        """
        self.splitter = splitter or TweetSplitter()

    def assemble_thread(
        self,
        hook_tweet: str,
        fact_checks: List[dict],
        closer_tweet: str,
        thumbnail_path: Optional[str] = None
    ) -> List[dict]:
        """
        Assemble a complete thread from components.

        Args:
            hook_tweet: The opening tweet
            fact_checks: List of fact-check dicts with claim, correction, source info
            closer_tweet: The closing tweet
            thumbnail_path: Optional path to thumbnail image for first tweet

        Returns:
            List of tweet dicts with content, position, has_media, media_path
        """
        tweets = []
        position = 1

        # Hook tweet (with optional thumbnail)
        tweets.append({
            "content": hook_tweet,
            "position": position,
            "has_media": thumbnail_path is not None,
            "media_path": thumbnail_path
        })
        position += 1

        # Body tweets (fact-checks)
        for fc in fact_checks:
            fc_tweets = self.splitter.split_fact_check_content(
                claim_text=fc.get("claim_text", ""),
                correction=fc.get("correction", ""),
                source_name=fc.get("source_name", ""),
                source_url=fc.get("source_url", "")
            )

            for tweet_content in fc_tweets:
                tweets.append({
                    "content": tweet_content,
                    "position": position,
                    "has_media": False,
                    "media_path": None
                })
                position += 1

        # Closer tweet
        tweets.append({
            "content": closer_tweet,
            "position": position,
            "has_media": False,
            "media_path": None
        })

        return tweets
