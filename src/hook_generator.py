#!/usr/bin/env python3
"""
Hook Variant Generator - Create multiple engaging openings for tweets.
Generates 5 different hook patterns optimized for engagement and A/B testing.
"""

import logging
import os
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from openai import OpenAI

logger = logging.getLogger(__name__)


class HookPattern(Enum):
    """Different hook patterns for tweet openings."""
    QUESTION = "question"  # "Ever wonder why...?"
    SHOCK = "shock"  # "This will blow your mind:"
    NUMBERS = "numbers"  # "97% of people don't know..."
    STORY = "story"  # "Here's what happened when..."
    CONTRARIAN = "contrarian"  # "Everything you know about X is wrong"
    CURIOSITY_GAP = "curiosity_gap"  # "The secret to X that nobody talks about"
    DIRECT = "direct"  # Straight to the point
    EMOTIONAL = "emotional"  # "I was shocked to discover..."


@dataclass
class HookVariant:
    """A generated hook variant."""
    pattern: HookPattern
    text: str
    full_tweet: str
    engagement_score: float  # Predicted engagement (0.0-1.0)
    character_count: int
    contains_emoji: bool
    ab_test_id: Optional[str] = None


class HookGenerator:
    """
    Generate multiple hook variants for A/B testing.

    Features:
    - 5+ different hook patterns
    - AI-generated variations
    - Engagement score prediction
    - A/B testing support
    - Template fallback for offline use
    """

    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """Initialize hook generator."""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        self.client = None
        if self.openai_api_key and self.openai_api_key != "your_openai_api_key_here":
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
                logger.info("✅ Hook generator initialized with AI")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize OpenAI: {e}")

        # Template-based hooks for fallback
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[HookPattern, List[str]]:
        """Load template hooks for each pattern."""
        return {
            HookPattern.QUESTION: [
                "Ever wonder {topic}?",
                "What if {topic}?",
                "Why do {topic}?",
                "Have you noticed {topic}?",
                "Curious about {topic}?"
            ],
            HookPattern.SHOCK: [
                "This will blow your mind:",
                "You won't believe this:",
                "Mind-blowing fact:",
                "This is wild:",
                "Prepare to be shocked:"
            ],
            HookPattern.NUMBERS: [
                "{number}% of people don't know this:",
                "Only {number} in {number2} people realize:",
                "After analyzing {number} cases:",
                "{number} proven ways to:",
                "The {number} most important things about:"
            ],
            HookPattern.STORY: [
                "Here's what happened when {action}:",
                "I tried {action} and:",
                "Last week, {event}. Here's what I learned:",
                "The moment I realized {realization}:",
                "A story about {topic}:"
            ],
            HookPattern.CONTRARIAN: [
                "Everyone's wrong about {topic}:",
                "The truth about {topic} that nobody tells you:",
                "Unpopular opinion: {opinion}",
                "Hot take: {take}",
                "Everything you know about {topic} is wrong."
            ],
            HookPattern.CURIOSITY_GAP: [
                "The secret to {goal} that nobody talks about:",
                "What {successful_people} don't want you to know:",
                "The hidden truth about {topic}:",
                "Here's what they're not telling you about {topic}:",
                "The one thing about {topic} everyone misses:"
            ],
            HookPattern.DIRECT: [
                "{topic}: A thread",
                "Let's talk about {topic}:",
                "Important: {topic}",
                "{topic} explained:",
                "Thread: {topic}"
            ],
            HookPattern.EMOTIONAL: [
                "I was shocked to discover {discovery}:",
                "This changed everything I knew about {topic}:",
                "I can't believe {realization}:",
                "This hit me hard: {insight}",
                "Honestly, {emotion} when I learned:"
            ]
        }

    def generate_variants(self,
                         insight_text: str,
                         podcast_name: str,
                         episode_title: str,
                         category: str = "general",
                         count: int = 5) -> List[HookVariant]:
        """
        Generate multiple hook variants for an insight.

        Args:
            insight_text: The main insight/content
            podcast_name: Source podcast
            episode_title: Episode title
            category: Insight category (quote, fact, takeaway, etc.)
            count: Number of variants to generate

        Returns:
            List of hook variants
        """

        logger.info(f"🎣 Generating {count} hook variants...")

        if self.client:
            # Use AI for better quality
            variants = self._generate_ai_variants(
                insight_text, podcast_name, episode_title, category, count
            )
        else:
            # Fallback to templates
            logger.info("Using template-based hooks (no AI available)")
            variants = self._generate_template_variants(
                insight_text, category, count
            )

        # Ensure we have the requested count
        if len(variants) < count:
            logger.warning(f"Only generated {len(variants)}/{count} variants")

        # Add A/B test IDs
        for i, variant in enumerate(variants):
            variant.ab_test_id = f"variant_{i+1}_{variant.pattern.value}"

        logger.info(f"✅ Generated {len(variants)} hook variants")

        return variants

    def _generate_ai_variants(self,
                             insight_text: str,
                             podcast_name: str,
                             episode_title: str,
                             category: str,
                             count: int) -> List[HookVariant]:
        """Generate hook variants using AI."""

        prompt = f"""You are an expert at writing viral social media hooks.

Content to promote:
"{insight_text}"

Source: {podcast_name} - {episode_title}
Category: {category}

Generate {count} different tweet opening hooks using these patterns:
1. QUESTION: Start with an engaging question
2. SHOCK: Create curiosity/surprise
3. NUMBERS: Use statistics or data
4. STORY: Begin with a narrative
5. CONTRARIAN: Challenge common beliefs

Requirements:
- Each hook should be 1-2 sentences max
- Optimized for Twitter engagement
- Diverse in approach
- Appropriate emoji (1-2) if it fits
- Lead naturally into the insight

Return as JSON array:
[
  {{
    "pattern": "question",
    "hook": "Ever wonder why...",
    "full_tweet": "Ever wonder why...? [insight continues]",
    "engagement_score": 0.85,
    "reasoning": "Questions drive engagement..."
  }},
  ...
]

Return ONLY the JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a viral content expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher creativity for hooks
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Handle both formats
            if isinstance(result, list):
                hooks_data = result
            elif "hooks" in result:
                hooks_data = result["hooks"]
            else:
                hooks_data = []

            variants = []

            for item in hooks_data[:count]:
                pattern_str = item.get("pattern", "direct")

                try:
                    pattern = HookPattern(pattern_str)
                except ValueError:
                    pattern = HookPattern.DIRECT

                full_tweet = item["full_tweet"]

                variants.append(
                    HookVariant(
                        pattern=pattern,
                        text=item["hook"],
                        full_tweet=full_tweet,
                        engagement_score=item.get("engagement_score", 0.7),
                        character_count=len(full_tweet),
                        contains_emoji=self._has_emoji(item["hook"])
                    )
                )

            return variants

        except Exception as e:
            logger.error(f"AI hook generation failed: {e}")
            # Fallback to templates
            return self._generate_template_variants(insight_text, category, count)

    def _generate_template_variants(self,
                                    insight_text: str,
                                    category: str,
                                    count: int) -> List[HookVariant]:
        """Generate hook variants using templates."""

        patterns = [
            HookPattern.QUESTION,
            HookPattern.SHOCK,
            HookPattern.CONTRARIAN,
            HookPattern.CURIOSITY_GAP,
            HookPattern.DIRECT
        ]

        variants = []

        for i, pattern in enumerate(patterns[:count]):
            templates = self.templates[pattern]
            template = random.choice(templates)

            # Simple variable filling
            hook_text = template.replace("{topic}", "this")
            hook_text = hook_text.replace("{action}", "something new")
            hook_text = hook_text.replace("{number}", str(random.choice([3, 5, 7, 10, 90, 95])))

            full_tweet = f"{hook_text}\n\n{insight_text}"

            variants.append(
                HookVariant(
                    pattern=pattern,
                    text=hook_text,
                    full_tweet=full_tweet,
                    engagement_score=0.6 + (random.random() * 0.2),  # 0.6-0.8
                    character_count=len(full_tweet),
                    contains_emoji=False
                )
            )

        return variants

    def _has_emoji(self, text: str) -> bool:
        """Check if text contains emoji."""
        # Simple emoji detection
        emoji_ranges = [
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
            (0x1F680, 0x1F6FF),  # Transport and Map
            (0x2600, 0x26FF),    # Misc symbols
            (0x2700, 0x27BF),    # Dingbats
        ]

        for char in text:
            code = ord(char)
            for start, end in emoji_ranges:
                if start <= code <= end:
                    return True

        return False

    def rank_variants_by_predicted_engagement(self,
                                              variants: List[HookVariant]) -> List[HookVariant]:
        """
        Rank hook variants by predicted engagement.

        Factors:
        - AI engagement score
        - Pattern performance (some patterns work better)
        - Length (optimal range)
        - Emoji usage
        """

        # Pattern performance weights (based on typical engagement)
        pattern_weights = {
            HookPattern.QUESTION: 1.2,
            HookPattern.NUMBERS: 1.15,
            HookPattern.CONTRARIAN: 1.1,
            HookPattern.SHOCK: 1.05,
            HookPattern.CURIOSITY_GAP: 1.1,
            HookPattern.STORY: 1.0,
            HookPattern.EMOTIONAL: 1.05,
            HookPattern.DIRECT: 0.9
        }

        def score_variant(variant: HookVariant) -> float:
            score = variant.engagement_score

            # Apply pattern weight
            score *= pattern_weights.get(variant.pattern, 1.0)

            # Length penalty (too short or too long)
            if variant.character_count < 100:
                score *= 0.9
            elif variant.character_count > 260:
                score *= 0.85

            # Emoji bonus
            if variant.contains_emoji:
                score *= 1.05

            return score

        # Sort by computed score
        ranked = sorted(variants, key=score_variant, reverse=True)

        return ranked

    def generate_ab_test_set(self,
                            insight_text: str,
                            podcast_name: str,
                            episode_title: str,
                            category: str = "general") -> Dict[str, Any]:
        """
        Generate a complete A/B test set with variants and metadata.

        Returns:
            Dictionary with variants, control, and test configuration
        """

        variants = self.generate_variants(
            insight_text, podcast_name, episode_title, category, count=5
        )

        # Rank by predicted performance
        ranked_variants = self.rank_variants_by_predicted_engagement(variants)

        # Top variant is the control
        control = ranked_variants[0]

        # Others are test variants
        test_variants = ranked_variants[1:]

        return {
            "control": {
                "id": control.ab_test_id,
                "text": control.full_tweet,
                "pattern": control.pattern.value,
                "engagement_score": control.engagement_score
            },
            "test_variants": [
                {
                    "id": v.ab_test_id,
                    "text": v.full_tweet,
                    "pattern": v.pattern.value,
                    "engagement_score": v.engagement_score
                }
                for v in test_variants
            ],
            "metadata": {
                "podcast_name": podcast_name,
                "episode_title": episode_title,
                "category": category,
                "total_variants": len(variants)
            }
        }


def main():
    """CLI for testing hook generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Hook Variant Generator")
    parser.add_argument("--test", action="store_true", help="Run test generation")
    parser.add_argument("--insight", type=str, help="Insight text to test")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generator = HookGenerator()

    if args.test or args.insight:
        insight = args.insight or "The secret to startup success isn't what you think. It's not about having the perfect idea or raising millions in VC. It's about solving a real problem and refusing to quit."

        variants = generator.generate_variants(
            insight_text=insight,
            podcast_name="Test Podcast",
            episode_title="Startup Secrets",
            category="hot_take",
            count=5
        )

        print(f"\n🎣 Generated {len(variants)} hook variants:\n")
        print("=" * 60)

        for i, variant in enumerate(variants, 1):
            print(f"\n{i}. Pattern: {variant.pattern.value.upper()}")
            print(f"   Engagement Score: {variant.engagement_score:.2f}")
            print(f"   Characters: {variant.character_count}")
            print(f"   Has Emoji: {variant.contains_emoji}")
            print(f"\n   Full Tweet:")
            print(f"   {variant.full_tweet}")
            print("-" * 60)

        # Show A/B test set
        print("\n\n📊 A/B TEST CONFIGURATION:\n")
        ab_test = generator.generate_ab_test_set(insight, "Test Podcast", "Startup Secrets")
        print(json.dumps(ab_test, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
