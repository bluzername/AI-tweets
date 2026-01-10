"""
Claim Extractor - Extracts both high-level insights and specific factual claims from transcripts.

This module extends the existing highlight extraction to also identify
specific verifiable factual claims for fact-checking.
"""

import logging
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .ai_analyzer import PodcastHighlight
from .unicode_utils import normalize_json_response

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of factual claims that can be verified."""
    STATISTIC = "statistic"          # "70% of Americans..."
    HISTORICAL = "historical"        # "In 1987, the government..."
    SCIENTIFIC = "scientific"        # "Studies show that..."
    ATTRIBUTION = "attribution"      # "Einstein said..."
    MEDICAL = "medical"              # "Vitamin X cures..."
    ECONOMIC = "economic"            # "The economy grew by..."
    GENERAL_FACT = "general_fact"    # Other verifiable facts


@dataclass
class FactualClaim:
    """Represents a specific factual claim extracted from a podcast."""
    claim_text: str                           # The verbatim or paraphrased claim
    speaker: Optional[str] = None             # Who said it (if identifiable)
    context: str = ""                         # Surrounding context (1-2 sentences)
    claim_type: ClaimType = ClaimType.GENERAL_FACT
    check_worthiness: float = 0.0             # 0.0-1.0 score from LLM filter
    timestamp: Optional[str] = None           # Approximate location in episode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "speaker": self.speaker,
            "context": self.context,
            "claim_type": self.claim_type.value,
            "check_worthiness": self.check_worthiness,
            "timestamp": self.timestamp
        }


@dataclass
class ExtractedClaims:
    """Container for both insights and factual claims extracted from an episode."""
    insights: List[PodcastHighlight] = field(default_factory=list)
    factual_claims: List[FactualClaim] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insights": [i.to_dict() for i in self.insights],
            "factual_claims": [c.to_dict() for c in self.factual_claims]
        }


class ClaimExtractor:
    """
    Extracts both high-level insights and specific factual claims from podcast transcripts.

    Uses DeepSeek via OpenRouter for cost-effective extraction.
    """

    def __init__(
        self,
        openrouter_api_key: str,
        model: str = "deepseek/deepseek-chat",
        checkworthy_threshold: float = 0.5
    ):
        """
        Initialize claim extractor.

        Args:
            openrouter_api_key: OpenRouter API key
            model: Model to use for extraction (default: DeepSeek)
            checkworthy_threshold: Minimum check-worthiness score to include claim
        """
        self.api_key = openrouter_api_key
        self.model = model
        self.checkworthy_threshold = checkworthy_threshold
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        logger.info(f"ClaimExtractor initialized with model: {model}")

    def _call_llm(self, messages: List[Dict], temperature: float = 0.3) -> Optional[str]:
        """Make a call to the LLM via OpenRouter."""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/ai-tweets",
                    "X-Title": "PodDebunker Claim Extractor",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"}
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def extract_all(
        self,
        transcription: str,
        podcast_name: str,
        episode_title: str,
        max_insights: int = 3,
        max_claims: int = 10
    ) -> ExtractedClaims:
        """
        Extract both insights and factual claims from a podcast transcript.

        Args:
            transcription: Full text transcription
            podcast_name: Name of the podcast
            episode_title: Title of the episode
            max_insights: Maximum number of insights to extract
            max_claims: Maximum number of factual claims to extract

        Returns:
            ExtractedClaims object containing both insights and claims
        """
        logger.info(f"Extracting claims from: {podcast_name} - {episode_title}")

        # Extract insights (high-level TLDR style)
        insights = self._extract_insights(
            transcription, podcast_name, episode_title, max_insights
        )

        # Extract specific factual claims for verification
        factual_claims = self._extract_factual_claims(
            transcription, podcast_name, episode_title, max_claims
        )

        # Filter claims by check-worthiness threshold
        filtered_claims = [
            c for c in factual_claims
            if c.check_worthiness >= self.checkworthy_threshold
        ]

        logger.info(
            f"Extracted {len(insights)} insights and "
            f"{len(filtered_claims)}/{len(factual_claims)} check-worthy claims"
        )

        return ExtractedClaims(
            insights=insights,
            factual_claims=filtered_claims
        )

    def _extract_insights(
        self,
        transcription: str,
        podcast_name: str,
        episode_title: str,
        max_insights: int
    ) -> List[PodcastHighlight]:
        """Extract high-level insights (similar to existing AIAnalyzer)."""

        system_prompt = """You are an expert content curator extracting viral-worthy insights
from podcast episodes. Focus on:
- Counterintuitive insights
- Practical, actionable advice
- Memorable quotes or stories
- Breakthrough moments or revelations
- Controversial or thought-provoking ideas

Each insight should be tweet-worthy and spark engagement."""

        user_prompt = f"""Analyze this podcast transcription and extract the TOP {max_insights}
most insightful, actionable highlights.

Podcast: {podcast_name}
Episode: {episode_title}

Transcription (first 8000 chars):
{transcription[:8000]}

For each highlight, provide:
1. title: A catchy title (5-10 words)
2. insight: The core insight (2-3 sentences)
3. actionable_advice: Practical takeaway
4. relevance_score: 0-1 (how universally applicable)
5. tweet_potential: 0-1 (viral potential on X)

Return as JSON:
{{"highlights": [{{"title": "", "insight": "", "actionable_advice": "", "relevance_score": 0.0, "tweet_potential": 0.0}}]}}
"""

        response = self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.7)

        if not response:
            return []

        try:
            data = normalize_json_response(json.loads(response))
            highlights_data = data.get("highlights", data if isinstance(data, list) else [])

            highlights = []
            for item in highlights_data[:max_insights]:
                highlights.append(PodcastHighlight(
                    title=item.get("title", ""),
                    insight=item.get("insight", ""),
                    actionable_advice=item.get("actionable_advice", ""),
                    relevance_score=float(item.get("relevance_score", 0.5)),
                    tweet_potential=float(item.get("tweet_potential", 0.5))
                ))

            return highlights

        except Exception as e:
            logger.error(f"Failed to parse insights: {e}")
            return []

    def _extract_factual_claims(
        self,
        transcription: str,
        podcast_name: str,
        episode_title: str,
        max_claims: int
    ) -> List[FactualClaim]:
        """Extract specific factual claims that can be verified."""

        system_prompt = """You are a fact-checking analyst. Your job is to identify specific
FACTUAL CLAIMS made in podcast transcripts that can be verified against external sources.

Focus on extracting:
- Statistics and numerical claims ("70% of...", "Studies show...")
- Historical claims ("In 1995, the government...", "X was the first to...")
- Scientific claims ("Research proves...", "Scientists discovered...")
- Attribution claims ("Einstein said...", "According to Harvard...")
- Medical/health claims ("This supplement cures...", "Doctors recommend...")
- Economic claims ("The market crashed because...", "GDP grew by...")

DO NOT extract:
- Personal opinions ("I think...", "In my view...")
- Future predictions ("This will happen...")
- Subjective assessments ("This is the best...")
- Obvious facts everyone knows
- Jokes or sarcasm

For each claim, assess its CHECK-WORTHINESS (0-1):
- 1.0: Highly specific, verifiable, potentially false/misleading
- 0.7: Verifiable fact that could use confirmation
- 0.4: Vague or difficult to verify
- 0.0: Opinion or unverifiable statement"""

        user_prompt = f"""Extract all specific factual claims from this podcast transcript.

Podcast: {podcast_name}
Episode: {episode_title}

Transcription:
{transcription[:12000]}

Return as JSON with up to {max_claims} claims:
{{
    "claims": [
        {{
            "claim_text": "The exact or paraphrased factual claim",
            "speaker": "Name if identifiable, else null",
            "context": "1-2 sentences of surrounding context",
            "claim_type": "statistic|historical|scientific|attribution|medical|economic|general_fact",
            "check_worthiness": 0.0-1.0,
            "timestamp": "approximate location if known, else null"
        }}
    ]
}}

Focus on claims that could potentially be FALSE or MISLEADING - these are most valuable to fact-check.
"""

        response = self._call_llm([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.2)

        if not response:
            return []

        try:
            data = normalize_json_response(json.loads(response))
            claims_data = data.get("claims", data if isinstance(data, list) else [])

            claims = []
            for item in claims_data[:max_claims]:
                claim_type_str = item.get("claim_type", "general_fact").lower()
                try:
                    claim_type = ClaimType(claim_type_str)
                except ValueError:
                    claim_type = ClaimType.GENERAL_FACT

                claims.append(FactualClaim(
                    claim_text=item.get("claim_text", ""),
                    speaker=item.get("speaker"),
                    context=item.get("context", ""),
                    claim_type=claim_type,
                    check_worthiness=float(item.get("check_worthiness", 0.5)),
                    timestamp=item.get("timestamp")
                ))

            # Sort by check-worthiness (most checkworthy first)
            claims.sort(key=lambda x: x.check_worthiness, reverse=True)

            return claims

        except Exception as e:
            logger.error(f"Failed to parse factual claims: {e}")
            return []

    def assess_checkworthiness(self, claim: FactualClaim) -> float:
        """
        Re-assess the check-worthiness of a single claim.

        Useful for filtering claims or getting a second opinion.
        """
        prompt = f"""Assess the CHECK-WORTHINESS of this claim on a scale of 0-1.

CLAIM: "{claim.claim_text}"
CONTEXT: "{claim.context}"
TYPE: {claim.claim_type.value}

Check-worthiness criteria:
- Is this a verifiable factual statement (not opinion)?
- Is this specific enough to fact-check?
- Could this claim potentially be false or misleading?
- Would verifying this claim be valuable to the public?

Return JSON: {{"check_worthiness": 0.0-1.0, "reasoning": "brief explanation"}}
"""

        response = self._call_llm([
            {"role": "user", "content": prompt}
        ], temperature=0.1)

        if not response:
            return claim.check_worthiness

        try:
            data = json.loads(response)
            return float(data.get("check_worthiness", claim.check_worthiness))
        except:
            return claim.check_worthiness
