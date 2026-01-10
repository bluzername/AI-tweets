#!/usr/bin/env python3
"""
Content Validator
Validates tweet threads before scheduling to ensure quality.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ContentValidator:
    """
    Validates tweet content for quality and correctness.

    Checks:
    - Thread structure (6 tweets)
    - Tweet lengths
    - Handle attribution
    - Content quality
    - No clickbait patterns
    - No filler words
    """

    def __init__(self):
        """Initialize content validator."""
        # Clickbait patterns to flag
        self.clickbait_patterns = [
            r'I used to .* until I learned',
            r"you won't believe",
            r"this will blow your mind",
            r"shocking truth",
            r"doctors hate",
            r"one weird trick",
            r"what happened next",
        ]

        # Filler words to flag
        self.filler_words = [
            'um', 'uh', 'you know', 'like,', 'basically', 'literally',
            'actually,', 'sort of', 'kind of', 'i mean'
        ]

        logger.info("ContentValidator initialized")

    def strip_filler_words(self, text: str) -> str:
        """
        Remove filler words from text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text with filler words removed
        """
        cleaned = text
        for filler in self.filler_words:
            # Handle filler words with punctuation (like "like,", "actually,")
            # Use word boundaries but allow trailing punctuation
            filler_clean = filler.rstrip(',.!?')
            pattern = re.compile(r'\b' + re.escape(filler_clean) + r'\b[,]?\s*', re.IGNORECASE)
            cleaned = pattern.sub('', cleaned)

        # Clean up extra spaces and punctuation issues
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned)  # Space before punctuation
        cleaned = re.sub(r'([.,!?])\s*\1+', r'\1', cleaned)  # Duplicate punctuation
        cleaned = re.sub(r'^\s*[,]\s*', '', cleaned)  # Leading comma
        cleaned = cleaned.strip()

        return cleaned

    def validate_thread(self, thread: List[str]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete thread and auto-clean filler words.

        Args:
            thread: List of tweet strings

        Returns:
            Tuple of (is_valid, list_of_errors, cleaned_thread)
        """
        errors = []

        # Auto-clean filler words from all tweets
        cleaned_thread = [self.strip_filler_words(tweet) for tweet in thread]

        # Check 1: Must be exactly 6 tweets
        if len(cleaned_thread) != 6:
            errors.append(f"Thread has {len(cleaned_thread)} tweets, needs exactly 6")
            return False, errors, cleaned_thread

        # Check 2: First tweet must have attribution
        if not self._has_attribution(cleaned_thread[0]):
            errors.append("Tweet 1 missing @handle attribution or podcast credit")

        # Check 3: Validate each tweet individually (using cleaned versions)
        for i, tweet in enumerate(cleaned_thread):
            tweet_num = i + 1
            tweet_errors = self._validate_tweet(tweet, tweet_num)
            errors.extend(tweet_errors)

        # Check 4: Ensure educational numbering in tweets 2-6
        for i in range(1, 6):
            if not self._has_number_emoji(cleaned_thread[i]):
                errors.append(f"Tweet {i+1} missing number emoji (1️⃣-5️⃣)")

        is_valid = len(errors) == 0
        return is_valid, errors, cleaned_thread

    def _validate_tweet(self, tweet: str, tweet_num: int) -> List[str]:
        """
        Validate an individual tweet.

        Args:
            tweet: Tweet text
            tweet_num: Tweet number in thread (1-6)

        Returns:
            List of error messages
        """
        errors = []

        # Check length
        if len(tweet) > 280:
            errors.append(f"Tweet {tweet_num} exceeds 280 chars ({len(tweet)})")

        # Check for empty content
        if not tweet.strip():
            errors.append(f"Tweet {tweet_num} is empty")

        # Check for clickbait patterns
        for pattern in self.clickbait_patterns:
            if re.search(pattern, tweet, re.IGNORECASE):
                errors.append(f"Tweet {tweet_num} contains clickbait pattern: {pattern}")

        # Note: Filler words are now auto-stripped in validate_thread(), so no check needed here

        # Check for incomplete sentences (tweet 1 gets a pass)
        if tweet_num > 1 and not self._is_complete_sentence(tweet):
            errors.append(f"Tweet {tweet_num} appears to be an incomplete sentence")

        return errors

    def _has_attribution(self, tweet: str) -> bool:
        """
        Check if tweet has proper attribution.

        Args:
            tweet: Tweet text

        Returns:
            True if has @handle or podcast credit
        """
        # Has @handle
        if '@' in tweet:
            return True

        # Has podcast emoji and credit phrase
        if '🎙️' in tweet and any(word in tweet.lower() for word in ['from', 'with', 'featuring']):
            return True

        return False

    def _has_number_emoji(self, tweet: str) -> bool:
        """
        Check if tweet has a number emoji (1️⃣-5️⃣).

        Args:
            tweet: Tweet text

        Returns:
            True if has number emoji
        """
        number_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣']
        return any(emoji in tweet for emoji in number_emojis)

    def _is_complete_sentence(self, tweet: str) -> bool:
        """
        Check if tweet appears to be a complete sentence.

        Args:
            tweet: Tweet text

        Returns:
            True if appears complete
        """
        # Remove number emoji and numbering
        text = re.sub(r'[1-5]️⃣\s*', '', tweet)
        text = re.sub(r'^\d+/\s*', '', text)
        text = text.strip()

        # Check minimum length
        if len(text) < 10:
            return False

        # Check for sentence-ending punctuation or emoji
        if text and text[-1] in '.!?':
            return True

        # Check if it's a statement (has verb-like structure)
        # This is a simple heuristic
        words = text.split()
        if len(words) >= 3:  # Minimum for a sentence
            return True

        return False

    def validate_content_quality(self, tweet: str) -> Tuple[bool, str]:
        """
        Check content quality of a single tweet.

        Args:
            tweet: Tweet text

        Returns:
            Tuple of (is_quality, feedback_message)
        """
        # Check for substance
        words = tweet.split()
        if len(words) < 5:
            return False, "Tweet too short, lacks substance"

        # Check for all caps (yelling)
        if tweet.isupper():
            return False, "Tweet is all caps"

        # Check for excessive punctuation
        if tweet.count('!') > 2 or tweet.count('?') > 2:
            return False, "Excessive punctuation"

        # Check for excessive emoji
        emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', tweet))
        if emoji_count > 5:
            return False, "Too many emojis"

        return True, "Quality check passed"

    def get_validation_summary(self, thread: List[str]) -> str:
        """
        Get a human-readable validation summary.

        Args:
            thread: List of tweet strings

        Returns:
            Summary string
        """
        is_valid, errors = self.validate_thread(thread)

        if is_valid:
            return "✅ Thread validation passed - ready to schedule"

        summary = f"❌ Thread validation failed with {len(errors)} errors:\n"
        for error in errors:
            summary += f"  - {error}\n"

        return summary
