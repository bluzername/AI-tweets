#!/usr/bin/env python3
"""
Multi-Model Insight Extractor - Ensemble AI Analysis
Uses GPT-4, Claude (Anthropic), and Gemini (Google) for consensus-based insight extraction.

Supports two modes:
1. OpenRouter: Single API key for all models (recommended)
2. Direct APIs: Separate keys for OpenAI, Anthropic, Google
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class AIModel(Enum):
    """Available AI models for ensemble."""
    # Direct API model names
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"
    CLAUDE_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_OPUS = "claude-3-opus-20240229"
    GEMINI_PRO = "gemini-pro"

    # OpenRouter model names (when using USE_OPENROUTER=true)
    OPENROUTER_GPT4_TURBO = "openai/gpt-4-turbo-preview"
    OPENROUTER_CLAUDE_SONNET = "anthropic/claude-3-sonnet"
    OPENROUTER_GEMINI_PRO = "google/gemini-pro-1.5"


@dataclass
class ModelInsight:
    """Insight extracted by a single model."""
    text: str
    category: str  # quote, fact, takeaway, hot_take, story
    confidence: float  # 0.0-1.0
    viral_score: float  # 0.0-1.0
    model: str
    reasoning: Optional[str] = None


@dataclass
class ConsensusInsight:
    """Insight agreed upon by multiple models."""
    text: str
    category: str
    viral_score: float
    confidence: float
    supporting_models: List[str]
    model_count: int
    variations: List[str]  # Different phrasings from different models


class MultiModelAnalyzer:
    """
    Ensemble AI analyzer using multiple models for better insights.

    Features:
    - Parallel extraction from GPT-4, Claude, and Gemini
    - Consensus voting on best insights
    - Deduplication of similar insights
    - Confidence scoring
    - Fallback chain if models unavailable
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 openrouter_api_key: Optional[str] = None,
                 use_openrouter: Optional[bool] = None,
                 enabled_models: Optional[List[AIModel]] = None):
        """
        Initialize multi-model analyzer.

        Args:
            openai_api_key: OpenAI API key (direct mode)
            anthropic_api_key: Anthropic API key (direct mode)
            google_api_key: Google AI API key (direct mode)
            openrouter_api_key: OpenRouter API key (unified mode)
            use_openrouter: Force OpenRouter mode (auto-detected if not specified)
            enabled_models: List of models to use (defaults to all available)
        """

        # Check if using OpenRouter
        self.use_openrouter = use_openrouter
        if self.use_openrouter is None:
            # Auto-detect: use OpenRouter if key is set or USE_OPENROUTER=true
            openrouter_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            use_openrouter_env = os.getenv("USE_OPENROUTER", "false").lower() == "true"
            self.use_openrouter = bool(openrouter_key) or use_openrouter_env

        if self.use_openrouter:
            logger.info("🔀 Using OpenRouter for unified AI model access")
            self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            # Direct API keys not used in OpenRouter mode
            self.openai_api_key = None
            self.anthropic_api_key = None
            self.google_api_key = None
        else:
            logger.info("🔑 Using direct API keys for each model")
            self.openrouter_api_key = None
            self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        self.openrouter_client = None

        self._init_clients()

        # Determine which models to use
        if enabled_models:
            self.enabled_models = enabled_models
        else:
            self.enabled_models = self._detect_available_models()

        logger.info(f"🤖 Multi-model analyzer initialized with: {[m.value for m in self.enabled_models]}")

    def _init_clients(self):
        """Initialize API clients for available models."""

        if self.use_openrouter:
            # OpenRouter mode: Single client for all models
            if self.openrouter_api_key and self.openrouter_api_key != "your_openrouter_api_key_here":
                try:
                    from openai import OpenAI
                    self.openrouter_client = OpenAI(
                        api_key=self.openrouter_api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
                    logger.info("✅ OpenRouter client initialized (unified access to all models)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize OpenRouter: {e}")
        else:
            # Direct API mode: Separate clients for each provider
            # OpenAI (GPT-4)
            if self.openai_api_key and self.openai_api_key != "your_openai_api_key_here":
                try:
                    from openai import OpenAI
                    self.openai_client = OpenAI(api_key=self.openai_api_key)
                    logger.info("✅ OpenAI client initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize OpenAI: {e}")

            # Anthropic (Claude)
            if self.anthropic_api_key and self.anthropic_api_key != "your_anthropic_api_key_here":
                try:
                    from anthropic import Anthropic
                    self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
                    logger.info("✅ Anthropic client initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Anthropic: {e}")

            # Google (Gemini)
            if self.google_api_key and self.google_api_key != "your_google_api_key_here":
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.google_api_key)
                    self.google_client = genai
                    logger.info("✅ Google AI client initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Google AI: {e}")

    def _detect_available_models(self) -> List[AIModel]:
        """Detect which models are available based on API keys."""
        available = []

        if self.use_openrouter and self.openrouter_client:
            # OpenRouter provides access to all models with one key
            available.extend([
                AIModel.OPENROUTER_GPT4_TURBO,
                AIModel.OPENROUTER_CLAUDE_SONNET,
                AIModel.OPENROUTER_GEMINI_PRO
            ])
        else:
            # Direct API mode: check each provider
            if self.openai_client:
                available.append(AIModel.GPT4_TURBO)

            if self.anthropic_client:
                available.append(AIModel.CLAUDE_SONNET)

            if self.google_client:
                available.append(AIModel.GEMINI_PRO)

        if not available:
            logger.warning("⚠️ No AI models available! Add API keys to .env")

        return available

    def extract_insights_ensemble(self,
                                   transcription: str,
                                   podcast_name: str,
                                   episode_title: str,
                                   max_insights: int = 5) -> List[ConsensusInsight]:
        """
        Extract insights using ensemble of models with consensus voting.

        Args:
            transcription: Episode transcription text
            podcast_name: Name of podcast
            episode_title: Episode title
            max_insights: Maximum insights to return

        Returns:
            List of consensus insights ranked by agreement and quality
        """

        logger.info(f"🤖 Extracting insights with {len(self.enabled_models)} models...")

        # Extract insights from each model
        all_insights: List[ModelInsight] = []

        for model in self.enabled_models:
            try:
                insights = self._extract_with_model(
                    model, transcription, podcast_name, episode_title, max_insights
                )
                all_insights.extend(insights)
                logger.info(f"✅ {model.value}: Extracted {len(insights)} insights")
            except Exception as e:
                logger.error(f"❌ {model.value} failed: {e}")
                continue

        if not all_insights:
            logger.error("❌ No insights extracted from any model")
            return []

        # Find consensus insights
        consensus_insights = self._find_consensus(all_insights)

        # Rank by quality (model agreement + viral score)
        consensus_insights.sort(
            key=lambda x: (x.model_count, x.viral_score, x.confidence),
            reverse=True
        )

        logger.info(f"✅ Found {len(consensus_insights)} consensus insights")

        return consensus_insights[:max_insights]

    def _extract_with_model(self,
                            model: AIModel,
                            transcription: str,
                            podcast_name: str,
                            episode_title: str,
                            max_insights: int) -> List[ModelInsight]:
        """Extract insights using a specific model."""

        prompt = self._build_extraction_prompt(
            transcription, podcast_name, episode_title, max_insights
        )

        # OpenRouter models (use unified OpenAI-compatible API)
        if model in [AIModel.OPENROUTER_GPT4_TURBO, AIModel.OPENROUTER_CLAUDE_SONNET, AIModel.OPENROUTER_GEMINI_PRO]:
            return self._extract_openrouter(model, prompt)
        # Direct API models
        elif model in [AIModel.GPT4, AIModel.GPT4_TURBO]:
            return self._extract_openai(model, prompt)
        elif model in [AIModel.CLAUDE_SONNET, AIModel.CLAUDE_OPUS]:
            return self._extract_anthropic(model, prompt)
        elif model == AIModel.GEMINI_PRO:
            return self._extract_google(prompt)
        else:
            raise ValueError(f"Unknown model: {model}")

    def _build_extraction_prompt(self,
                                  transcription: str,
                                  podcast_name: str,
                                  episode_title: str,
                                  max_insights: int) -> str:
        """Build the extraction prompt for any model."""

        return f"""You are an expert at analyzing podcast content and extracting viral-worthy insights for social media.

Podcast: {podcast_name}
Episode: {episode_title}

Transcription:
{transcription[:8000]}  # Limit to avoid token limits

Your task: Extract the {max_insights} most compelling, tweet-worthy insights from this content.

For each insight, provide:
1. The insight text (1-2 sentences, punchy and engaging)
2. Category: quote | fact | takeaway | hot_take | story
3. Viral score (0.0-1.0): How likely to go viral
4. Confidence (0.0-1.0): How confident you are this is valuable
5. Reasoning: Why this is noteworthy (1 sentence)

Focus on:
- Surprising facts or statistics
- Contrarian or hot takes
- Actionable advice
- Compelling personal stories
- Powerful quotes
- Insights that make people think differently

Return as JSON array:
[
  {{
    "text": "The insight...",
    "category": "hot_take",
    "viral_score": 0.85,
    "confidence": 0.90,
    "reasoning": "This challenges conventional wisdom..."
  }},
  ...
]

Return ONLY the JSON array, no other text."""

    def _extract_openai(self, model: AIModel, prompt: str) -> List[ModelInsight]:
        """Extract insights using OpenAI (GPT-4)."""

        try:
            response = self.openai_client.chat.completions.create(
                model=model.value,
                messages=[
                    {"role": "system", "content": "You are an expert content analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Handle both array and object with array field
            if isinstance(result, list):
                insights_data = result
            elif "insights" in result:
                insights_data = result["insights"]
            else:
                insights_data = []

            return [
                ModelInsight(
                    text=item["text"],
                    category=item["category"],
                    viral_score=item["viral_score"],
                    confidence=item["confidence"],
                    reasoning=item.get("reasoning"),
                    model=model.value
                )
                for item in insights_data
            ]

        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return []

    def _extract_openrouter(self, model: AIModel, prompt: str) -> List[ModelInsight]:
        """Extract insights using OpenRouter (unified API for all models)."""

        try:
            response = self.openrouter_client.chat.completions.create(
                model=model.value,
                messages=[
                    {"role": "system", "content": "You are an expert content analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            content = response.choices[0].message.content

            # Parse JSON from response (some models wrap in markdown)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Handle both array and object with array field
            if isinstance(result, list):
                insights_data = result
            elif "insights" in result:
                insights_data = result["insights"]
            else:
                insights_data = []

            return [
                ModelInsight(
                    text=item["text"],
                    category=item["category"],
                    viral_score=item["viral_score"],
                    confidence=item["confidence"],
                    reasoning=item.get("reasoning"),
                    model=model.value
                )
                for item in insights_data
            ]

        except Exception as e:
            logger.error(f"OpenRouter extraction failed for {model.value}: {e}")
            return []

    def _extract_anthropic(self, model: AIModel, prompt: str) -> List[ModelInsight]:
        """Extract insights using Anthropic (Claude)."""

        try:
            response = self.anthropic_client.messages.create(
                model=model.value,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.content[0].text

            # Parse JSON from response
            # Claude sometimes wraps in markdown, so extract
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Handle both array and object with array field
            if isinstance(result, list):
                insights_data = result
            elif "insights" in result:
                insights_data = result["insights"]
            else:
                insights_data = []

            return [
                ModelInsight(
                    text=item["text"],
                    category=item["category"],
                    viral_score=item["viral_score"],
                    confidence=item["confidence"],
                    reasoning=item.get("reasoning"),
                    model=model.value
                )
                for item in insights_data
            ]

        except Exception as e:
            logger.error(f"Anthropic extraction failed: {e}")
            return []

    def _extract_google(self, prompt: str) -> List[ModelInsight]:
        """Extract insights using Google (Gemini)."""

        try:
            model = self.google_client.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)

            content = response.text

            # Parse JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Handle both array and object with array field
            if isinstance(result, list):
                insights_data = result
            elif "insights" in result:
                insights_data = result["insights"]
            else:
                insights_data = []

            return [
                ModelInsight(
                    text=item["text"],
                    category=item["category"],
                    viral_score=item["viral_score"],
                    confidence=item["confidence"],
                    reasoning=item.get("reasoning"),
                    model="gemini-pro"
                )
                for item in insights_data
            ]

        except Exception as e:
            logger.error(f"Google AI extraction failed: {e}")
            return []

    def _find_consensus(self, all_insights: List[ModelInsight]) -> List[ConsensusInsight]:
        """
        Find insights that multiple models agree on using semantic similarity.

        Uses simple text similarity for now. Could be enhanced with embeddings.
        """

        if len(all_insights) < 2:
            # Only one model - return all as consensus with count=1
            return [
                ConsensusInsight(
                    text=insight.text,
                    category=insight.category,
                    viral_score=insight.viral_score,
                    confidence=insight.confidence,
                    supporting_models=[insight.model],
                    model_count=1,
                    variations=[insight.text]
                )
                for insight in all_insights
            ]

        # Group similar insights
        clusters = []
        used = set()

        for i, insight1 in enumerate(all_insights):
            if i in used:
                continue

            cluster = [insight1]
            used.add(i)

            for j, insight2 in enumerate(all_insights[i+1:], start=i+1):
                if j in used:
                    continue

                # Check similarity
                if self._are_similar(insight1.text, insight2.text):
                    cluster.append(insight2)
                    used.add(j)

            clusters.append(cluster)

        # Convert clusters to consensus insights
        consensus_insights = []

        for cluster in clusters:
            # Pick the best phrasing (highest viral score)
            best = max(cluster, key=lambda x: x.viral_score)

            consensus_insights.append(
                ConsensusInsight(
                    text=best.text,
                    category=best.category,
                    viral_score=sum(i.viral_score for i in cluster) / len(cluster),
                    confidence=sum(i.confidence for i in cluster) / len(cluster),
                    supporting_models=[i.model for i in cluster],
                    model_count=len(cluster),
                    variations=[i.text for i in cluster]
                )
            )

        return consensus_insights

    def _are_similar(self, text1: str, text2: str, threshold: float = 0.6) -> bool:
        """
        Check if two insight texts are similar.

        Uses simple word overlap for now. Could be enhanced with embeddings.
        """

        # Normalize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return False

        similarity = intersection / union

        return similarity >= threshold

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about model usage and performance."""

        return {
            "enabled_models": [m.value for m in self.enabled_models],
            "available_clients": {
                "openai": self.openai_client is not None,
                "anthropic": self.anthropic_client is not None,
                "google": self.google_client is not None
            },
            "model_count": len(self.enabled_models)
        }


def main():
    """CLI for testing multi-model analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Model Insight Analyzer")
    parser.add_argument("--test", action="store_true", help="Run test extraction")
    parser.add_argument("--stats", action="store_true", help="Show model statistics")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    analyzer = MultiModelAnalyzer()

    if args.stats:
        stats = analyzer.get_model_statistics()
        print(json.dumps(stats, indent=2))

    elif args.test:
        test_transcript = """
        The secret to building a successful startup isn't what most people think.
        Everyone says you need venture capital and a perfect team from day one.
        But I've found that 90% of unicorn companies started with less than $10,000
        and a solo founder who just refused to quit. The real differentiator isn't
        money or team - it's obsessive focus on solving one problem really well.
        """

        insights = analyzer.extract_insights_ensemble(
            transcription=test_transcript,
            podcast_name="Test Podcast",
            episode_title="Startup Secrets",
            max_insights=3
        )

        print(f"\n✅ Extracted {len(insights)} consensus insights:\n")

        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight.text}")
            print(f"   Category: {insight.category}")
            print(f"   Viral Score: {insight.viral_score:.2f}")
            print(f"   Models: {', '.join(insight.supporting_models)}")
            print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
