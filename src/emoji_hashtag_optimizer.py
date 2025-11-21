#!/usr/bin/env python3
"""
Emoji and Hashtag Optimizer - Strategic emoji and hashtag selection.
Intelligently adds emojis and hashtags to maximize engagement without being spammy.
"""

import logging
import os
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class EmojiCategory(Enum):
    """Emoji categories for different contexts."""
    TECH = "tech"
    BUSINESS = "business"
    EDUCATION = "education"
    MOTIVATION = "motivation"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    GENERAL = "general"


@dataclass
class EmojiRecommendation:
    """Recommended emoji with context."""
    emoji: str
    position: str  # "start", "end", "inline"
    relevance_score: float
    context: str  # Why this emoji


@dataclass
class HashtagRecommendation:
    """Recommended hashtag with metadata."""
    hashtag: str
    relevance_score: float
    estimated_reach: str  # "high", "medium", "low"
    trending: bool


class EmojiHashtagOptimizer:
    """
    Optimize emoji and hashtag usage for better engagement.

    Features:
    - Context-aware emoji selection
    - Strategic emoji placement (not spammy)
    - Industry-specific hashtag libraries
    - Trending hashtag detection
    - Reach optimization
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize optimizer."""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Load emoji libraries
        self.emoji_library = self._load_emoji_library()

        # Load hashtag libraries
        self.hashtag_library = self._load_hashtag_library()

        # AI client for advanced recommendations
        self.client = None
        if self.openai_api_key and self.openai_api_key != "your_openai_api_key_here":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                logger.info("✅ Emoji/Hashtag optimizer initialized with AI")
            except Exception as e:
                logger.warning(f"⚠️ AI recommendations unavailable: {e}")

    def _load_emoji_library(self) -> Dict[EmojiCategory, Dict[str, str]]:
        """Load categorized emoji library."""
        return {
            EmojiCategory.TECH: {
                "rocket": "🚀",  # Launch, growth, startup
                "bulb": "💡",  # Ideas, innovation
                "computer": "💻",  # Tech, coding
                "robot": "🤖",  # AI, automation
                "gear": "⚙️",  # Engineering, systems
                "chart_up": "📈",  # Growth, metrics
                "zap": "⚡",  # Speed, energy
                "fire": "🔥",  # Hot, trending
                "brain": "🧠",  # Intelligence, thinking
                "target": "🎯",  # Goals, precision
            },
            EmojiCategory.BUSINESS: {
                "money_bag": "💰",  # Money, revenue
                "chart_up": "📈",  # Growth
                "handshake": "🤝",  # Partnership, deal
                "trophy": "🏆",  # Success, win
                "briefcase": "💼",  # Business, professional
                "target": "🎯",  # Goals
                "rocket": "🚀",  # Growth
                "crown": "👑",  # Leadership
                "gem": "💎",  # Value, premium
                "key": "🔑",  # Essential, unlock
            },
            EmojiCategory.EDUCATION: {
                "books": "📚",  # Learning, education
                "graduation": "🎓",  # Achievement
                "bulb": "💡",  # Learning
                "brain": "🧠",  # Knowledge
                "pencil": "✏️",  # Writing, notes
                "check": "✅",  # Correct, done
                "star": "⭐",  # Excellence
                "magnifying_glass": "🔍",  # Research, detail
                "microscope": "🔬",  # Science, deep dive
                "telescope": "🔭",  # Vision, future
            },
            EmojiCategory.MOTIVATION: {
                "fire": "🔥",  # Passion, energy
                "flex": "💪",  # Strength, power
                "rocket": "🚀",  # Growth
                "trophy": "🏆",  # Success
                "star": "⭐",  # Excellence
                "zap": "⚡",  # Energy
                "boom": "💥",  # Impact
                "sparkles": "✨",  # Magic, special
                "mountain": "⛰️",  # Challenge, achievement
                "sunrise": "🌅",  # New beginning
            },
            EmojiCategory.ENTERTAINMENT: {
                "popcorn": "🍿",  # Entertainment
                "movie": "🎬",  # Film, video
                "microphone": "🎤",  # Podcast, speaking
                "headphones": "🎧",  # Audio, listening
                "clapper": "🎬",  # Action, production
                "star": "⭐",  # Featured
                "fire": "🔥",  # Hot content
                "eyes": "👀",  # Watch, attention
                "mind_blown": "🤯",  # Shocking
                "laugh": "😂",  # Funny
            },
            EmojiCategory.SCIENCE: {
                "microscope": "🔬",  # Research
                "test_tube": "🧪",  # Experiment
                "dna": "🧬",  # Genetics, biology
                "atom": "⚛️",  # Physics, science
                "planet": "🌍",  # Global, earth
                "telescope": "🔭",  # Astronomy
                "brain": "🧠",  # Neuroscience
                "magnifying_glass": "🔍",  # Research
                "chart": "📊",  # Data, analysis
                "rocket": "🚀",  # Space, innovation
            },
            EmojiCategory.GENERAL: {
                "point_down": "👇",  # Direct attention
                "thread": "🧵",  # Thread indicator
                "fire": "🔥",  # Hot, trending
                "sparkles": "✨",  # Special, highlight
                "zap": "⚡",  # Quick, important
                "star": "⭐",  # Featured
                "check": "✅",  # Confirmed, done
                "warning": "⚠️",  # Important, caution
                "info": "ℹ️",  # Information
                "question": "❓",  # Question
            }
        }

    def _load_hashtag_library(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load categorized hashtag library with reach estimates."""
        return {
            "tech": [
                {"tag": "#Tech", "reach": "high", "specificity": "low"},
                {"tag": "#Technology", "reach": "high", "specificity": "low"},
                {"tag": "#AI", "reach": "high", "specificity": "medium"},
                {"tag": "#MachineLearning", "reach": "medium", "specificity": "medium"},
                {"tag": "#Startup", "reach": "high", "specificity": "medium"},
                {"tag": "#Innovation", "reach": "high", "specificity": "low"},
                {"tag": "#Coding", "reach": "medium", "specificity": "medium"},
                {"tag": "#Developer", "reach": "medium", "specificity": "medium"},
                {"tag": "#TechNews", "reach": "medium", "specificity": "medium"},
                {"tag": "#FutureTech", "reach": "low", "specificity": "medium"},
            ],
            "business": [
                {"tag": "#Business", "reach": "high", "specificity": "low"},
                {"tag": "#Entrepreneur", "reach": "high", "specificity": "medium"},
                {"tag": "#Leadership", "reach": "medium", "specificity": "medium"},
                {"tag": "#Marketing", "reach": "high", "specificity": "medium"},
                {"tag": "#Sales", "reach": "medium", "specificity": "medium"},
                {"tag": "#Productivity", "reach": "medium", "specificity": "medium"},
                {"tag": "#Success", "reach": "high", "specificity": "low"},
                {"tag": "#Mindset", "reach": "medium", "specificity": "medium"},
                {"tag": "#GrowthHacking", "reach": "low", "specificity": "high"},
                {"tag": "#BusinessTips", "reach": "medium", "specificity": "medium"},
            ],
            "education": [
                {"tag": "#Learning", "reach": "high", "specificity": "low"},
                {"tag": "#Education", "reach": "high", "specificity": "low"},
                {"tag": "#Knowledge", "reach": "medium", "specificity": "low"},
                {"tag": "#StudyTips", "reach": "medium", "specificity": "medium"},
                {"tag": "#SkillDevelopment", "reach": "low", "specificity": "medium"},
                {"tag": "#OnlineLearning", "reach": "medium", "specificity": "medium"},
                {"tag": "#PersonalGrowth", "reach": "high", "specificity": "medium"},
                {"tag": "#SelfImprovement", "reach": "medium", "specificity": "medium"},
            ],
            "podcast": [
                {"tag": "#Podcast", "reach": "high", "specificity": "low"},
                {"tag": "#Podcasts", "reach": "high", "specificity": "low"},
                {"tag": "#PodcastRecommendation", "reach": "medium", "specificity": "medium"},
                {"tag": "#Listen", "reach": "low", "specificity": "low"},
                {"tag": "#Audio", "reach": "low", "specificity": "low"},
                {"tag": "#PodcastClip", "reach": "low", "specificity": "high"},
                {"tag": "#PodcastHighlights", "reach": "low", "specificity": "high"},
            ],
            "general": [
                {"tag": "#Threadđź§µ", "reach": "medium", "specificity": "low"},
                {"tag": "#MustRead", "reach": "medium", "specificity": "low"},
                {"tag": "#Viral", "reach": "high", "specificity": "low"},
                {"tag": "#Trending", "reach": "medium", "specificity": "low"},
                {"tag": "#Insights", "reach": "medium", "specificity": "low"},
            ]
        }

    def recommend_emojis(self,
                        text: str,
                        category: EmojiCategory = EmojiCategory.GENERAL,
                        max_emojis: int = 2) -> List[EmojiRecommendation]:
        """
        Recommend emojis for given text.

        Args:
            text: Tweet text
            category: Content category
            max_emojis: Maximum emojis to recommend

        Returns:
            List of emoji recommendations
        """

        # Get relevant emoji set
        emoji_set = self.emoji_library.get(category, self.emoji_library[EmojiCategory.GENERAL])

        # Analyze text for keywords
        text_lower = text.lower()

        recommendations = []

        # Score each emoji by relevance
        for name, emoji in emoji_set.items():
            score = 0.0

            # Keyword matching (simple heuristic)
            keywords = {
                "rocket": ["launch", "start", "begin", "grow", "scale"],
                "bulb": ["idea", "innovation", "creative", "think"],
                "fire": ["hot", "trending", "popular", "viral"],
                "chart_up": ["grow", "increase", "improve", "scale", "revenue"],
                "brain": ["think", "intelligent", "smart", "mind"],
                "trophy": ["win", "success", "achieve", "goal"],
                "zap": ["fast", "quick", "speed", "instant"],
                "target": ["goal", "focus", "aim", "precision"],
            }

            if name in keywords:
                for keyword in keywords[name]:
                    if keyword in text_lower:
                        score += 0.3

            # Category relevance
            score += 0.5

            if score > 0:
                recommendations.append(
                    EmojiRecommendation(
                        emoji=emoji,
                        position="start" if len(recommendations) == 0 else "end",
                        relevance_score=score,
                        context=f"Matches {name} context"
                    )
                )

        # Sort by score
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)

        return recommendations[:max_emojis]

    def recommend_hashtags(self,
                          text: str,
                          podcast_name: Optional[str] = None,
                          categories: Optional[List[str]] = None,
                          max_hashtags: int = 3) -> List[HashtagRecommendation]:
        """
        Recommend hashtags for given text.

        Args:
            text: Tweet text
            podcast_name: Source podcast (can extract category)
            categories: Content categories
            max_hashtags: Maximum hashtags to recommend

        Returns:
            List of hashtag recommendations
        """

        if categories is None:
            categories = ["general", "podcast"]

        # Collect candidate hashtags
        candidates = []

        for category in categories:
            if category in self.hashtag_library:
                for hashtag_info in self.hashtag_library[category]:
                    candidates.append(hashtag_info)

        # Score by relevance
        text_lower = text.lower()

        recommendations = []

        for hashtag_info in candidates:
            tag = hashtag_info["tag"]
            tag_lower = tag.lower().replace("#", "")

            score = 0.0

            # Check if hashtag words appear in text
            if tag_lower in text_lower:
                score += 0.5

            # Reach bonus
            reach = hashtag_info["reach"]
            if reach == "high":
                score += 0.3
            elif reach == "medium":
                score += 0.2

            # Specificity (more specific is better for niche content)
            specificity = hashtag_info["specificity"]
            if specificity == "high":
                score += 0.2
            elif specificity == "medium":
                score += 0.1

            recommendations.append(
                HashtagRecommendation(
                    hashtag=tag,
                    relevance_score=score,
                    estimated_reach=reach,
                    trending=False  # Could integrate with Twitter API
                )
            )

        # Sort by score
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)

        # Diversify reach (mix high and medium reach)
        final_recommendations = []
        high_reach_added = 0
        medium_reach_added = 0

        for rec in recommendations:
            if len(final_recommendations) >= max_hashtags:
                break

            if rec.estimated_reach == "high" and high_reach_added < 2:
                final_recommendations.append(rec)
                high_reach_added += 1
            elif rec.estimated_reach == "medium" and medium_reach_added < 2:
                final_recommendations.append(rec)
                medium_reach_added += 1
            elif len(final_recommendations) < max_hashtags:
                final_recommendations.append(rec)

        return final_recommendations[:max_hashtags]

    def optimize_tweet(self,
                      text: str,
                      category: EmojiCategory = EmojiCategory.GENERAL,
                      content_categories: Optional[List[str]] = None,
                      max_emojis: int = 2,
                      max_hashtags: int = 3) -> Dict[str, Any]:
        """
        Optimize tweet with emojis and hashtags.

        Args:
            text: Original tweet text
            category: Emoji category
            content_categories: Content categories for hashtags
            max_emojis: Maximum emojis
            max_hashtags: Maximum hashtags

        Returns:
            Dictionary with optimized tweet and metadata
        """

        # Get recommendations
        emoji_recs = self.recommend_emojis(text, category, max_emojis)
        hashtag_recs = self.recommend_hashtags(text, None, content_categories, max_hashtags)

        # Build optimized tweet
        optimized = text

        # Add emojis
        start_emojis = [r.emoji for r in emoji_recs if r.position == "start"]
        end_emojis = [r.emoji for r in emoji_recs if r.position == "end"]

        if start_emojis:
            optimized = " ".join(start_emojis) + " " + optimized

        # Add hashtags at end
        if hashtag_recs:
            hashtags = " ".join([r.hashtag for r in hashtag_recs])
            optimized = optimized + "\n\n" + hashtags

        if end_emojis:
            optimized = optimized + " " + " ".join(end_emojis)

        return {
            "original": text,
            "optimized": optimized,
            "character_count": len(optimized),
            "emojis_added": [r.emoji for r in emoji_recs],
            "hashtags_added": [r.hashtag for r in hashtag_recs],
            "emoji_recommendations": [
                {
                    "emoji": r.emoji,
                    "position": r.position,
                    "score": r.relevance_score
                }
                for r in emoji_recs
            ],
            "hashtag_recommendations": [
                {
                    "hashtag": r.hashtag,
                    "reach": r.estimated_reach,
                    "score": r.relevance_score
                }
                for r in hashtag_recs
            ]
        }


def main():
    """CLI for emoji/hashtag optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Emoji and Hashtag Optimizer")
    parser.add_argument("--text", type=str, help="Tweet text to optimize")
    parser.add_argument("--category", type=str, default="general",
                       choices=["tech", "business", "education", "motivation", "entertainment", "science", "general"],
                       help="Content category")
    parser.add_argument("--max-emojis", type=int, default=2, help="Max emojis")
    parser.add_argument("--max-hashtags", type=int, default=3, help="Max hashtags")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    optimizer = EmojiHashtagOptimizer()

    if args.text:
        # Map string to enum
        category_map = {
            "tech": EmojiCategory.TECH,
            "business": EmojiCategory.BUSINESS,
            "education": EmojiCategory.EDUCATION,
            "motivation": EmojiCategory.MOTIVATION,
            "entertainment": EmojiCategory.ENTERTAINMENT,
            "science": EmojiCategory.SCIENCE,
            "general": EmojiCategory.GENERAL,
        }

        result = optimizer.optimize_tweet(
            text=args.text,
            category=category_map[args.category],
            content_categories=[args.category, "podcast"],
            max_emojis=args.max_emojis,
            max_hashtags=args.max_hashtags
        )

        print("\n" + "="*60)
        print("ORIGINAL:")
        print("="*60)
        print(result["original"])
        print("\n" + "="*60)
        print("OPTIMIZED:")
        print("="*60)
        print(result["optimized"])
        print("\n" + "="*60)
        print(f"Character count: {result['character_count']}/280")
        print(f"Emojis added: {', '.join(result['emojis_added'])}")
        print(f"Hashtags added: {', '.join(result['hashtags_added'])}")
        print("="*60)

    else:
        # Demo
        demo_text = "The secret to startup success isn't what you think. It's not about having perfect timing or lots of money. It's about solving a real problem and never giving up."

        result = optimizer.optimize_tweet(
            text=demo_text,
            category=EmojiCategory.BUSINESS,
            content_categories=["business", "tech"],
            max_emojis=2,
            max_hashtags=3
        )

        print("\nDEMO OPTIMIZATION:\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
