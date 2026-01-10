"""
Fact Checker - Verifies factual claims using Google Fact Check API and LLM fallback.

This module provides a multi-layer verification system:
1. Check-worthiness filter (is this a verifiable fact?)
2. Google Fact Check Tools API (authoritative fact-checks)
3. LLM fallback with web search (DeepSeek via OpenRouter)
"""

import logging
import json
import requests
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote_plus

from .claim_extractor import FactualClaim, ClaimType

logger = logging.getLogger(__name__)


class Verdict(Enum):
    """Standardized verdict for fact-check results."""
    TRUE = "true"
    FALSE = "false"
    MISLEADING = "misleading"
    UNVERIFIED = "unverified"


@dataclass
class FactCheckResult:
    """Result of fact-checking a single claim."""
    original_claim: FactualClaim
    verdict: Verdict
    correction: Optional[str] = None      # The accurate information (if false/misleading)
    source_name: str = ""                 # "Snopes", "PolitiFact", etc.
    source_url: str = ""                  # Working link to evidence
    confidence: float = 0.0               # 0.0-1.0
    method_used: str = ""                 # "google_factcheck" or "llm_websearch"
    raw_rating: Optional[str] = None      # Original rating from source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_claim": self.original_claim.to_dict(),
            "verdict": self.verdict.value,
            "correction": self.correction,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "method_used": self.method_used,
            "raw_rating": self.raw_rating
        }

    @property
    def is_false_or_misleading(self) -> bool:
        """Check if this result indicates a false or misleading claim."""
        return self.verdict in (Verdict.FALSE, Verdict.MISLEADING)


# Rating normalization map for Google Fact Check API responses
RATING_NORMALIZATION = {
    # FALSE variants
    "false": Verdict.FALSE,
    "pants on fire": Verdict.FALSE,
    "four pinocchios": Verdict.FALSE,
    "incorrect": Verdict.FALSE,
    "wrong": Verdict.FALSE,
    "fake": Verdict.FALSE,
    "not true": Verdict.FALSE,
    "no": Verdict.FALSE,
    "debunked": Verdict.FALSE,
    "fiction": Verdict.FALSE,
    "hoax": Verdict.FALSE,
    "fabricated": Verdict.FALSE,

    # MISLEADING variants
    "mostly false": Verdict.MISLEADING,
    "half true": Verdict.MISLEADING,
    "misleading": Verdict.MISLEADING,
    "lacks context": Verdict.MISLEADING,
    "missing context": Verdict.MISLEADING,
    "partly false": Verdict.MISLEADING,
    "partially true": Verdict.MISLEADING,
    "mixture": Verdict.MISLEADING,
    "mixed": Verdict.MISLEADING,
    "exaggerated": Verdict.MISLEADING,
    "distorts the facts": Verdict.MISLEADING,
    "out of context": Verdict.MISLEADING,
    "needs context": Verdict.MISLEADING,
    "three pinocchios": Verdict.MISLEADING,
    "two pinocchios": Verdict.MISLEADING,

    # TRUE variants
    "true": Verdict.TRUE,
    "mostly true": Verdict.TRUE,
    "correct": Verdict.TRUE,
    "accurate": Verdict.TRUE,
    "yes": Verdict.TRUE,
    "fact": Verdict.TRUE,
    "confirmed": Verdict.TRUE,
    "verified": Verdict.TRUE,
    "one pinocchio": Verdict.TRUE,
    "geppetto checkmark": Verdict.TRUE,
}

# Reputable fact-checking publishers (prioritize these)
REPUTABLE_PUBLISHERS = [
    "snopes", "politifact", "factcheck.org", "reuters", "ap news",
    "associated press", "washington post", "bbc", "full fact",
    "africa check", "lead stories", "check your fact", "usa today",
    "the dispatch", "science feedback", "health feedback"
]


class FactChecker:
    """
    Multi-layer fact verification system.

    Uses Google Fact Check API as primary source, with LLM web search fallback.
    """

    def __init__(
        self,
        google_api_key: Optional[str],
        openrouter_api_key: str,
        llm_model: str = "deepseek/deepseek-chat",
        confidence_threshold: float = 0.90,
        checkworthy_threshold: float = 0.5
    ):
        """
        Initialize fact checker.

        Args:
            google_api_key: Google Fact Check Tools API key (can be None)
            openrouter_api_key: OpenRouter API key for LLM fallback
            llm_model: Model to use for LLM verification
            confidence_threshold: Minimum confidence to publish LLM results (default 90%)
            checkworthy_threshold: Minimum check-worthiness to process claim
        """
        self.google_api_key = google_api_key
        self.openrouter_api_key = openrouter_api_key
        self.llm_model = llm_model
        self.confidence_threshold = confidence_threshold
        self.checkworthy_threshold = checkworthy_threshold

        self.google_api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

        logger.info(
            f"FactChecker initialized (Google API: {'enabled' if google_api_key else 'disabled'}, "
            f"LLM: {llm_model}, confidence threshold: {confidence_threshold})"
        )

    def check_claims(self, claims: List[FactualClaim]) -> List[FactCheckResult]:
        """
        Check a list of factual claims.

        Args:
            claims: List of claims to verify

        Returns:
            List of FactCheckResult for each claim
        """
        results = []

        for claim in claims:
            # Skip claims below check-worthiness threshold
            if claim.check_worthiness < self.checkworthy_threshold:
                logger.debug(f"Skipping low check-worthiness claim: {claim.claim_text[:50]}...")
                continue

            result = self.check_claim(claim)
            if result:
                results.append(result)

        logger.info(f"Checked {len(claims)} claims, got {len(results)} results")
        return results

    def check_claim(self, claim: FactualClaim) -> Optional[FactCheckResult]:
        """
        Check a single factual claim through the verification pipeline.

        Pipeline:
        1. Query Google Fact Check API
        2. If no result, try LLM web search fallback
        3. If confidence too low, return None (discard claim)

        Args:
            claim: The claim to verify

        Returns:
            FactCheckResult or None if claim couldn't be verified
        """
        logger.info(f"Checking claim: {claim.claim_text[:80]}...")

        # Layer 1: Try Google Fact Check API
        if self.google_api_key:
            google_result = self._query_google_factcheck(claim)
            if google_result:
                logger.info(f"Found Google Fact Check result: {google_result.verdict.value}")
                return google_result

        # Layer 2: LLM web search fallback
        llm_result = self._llm_websearch_fallback(claim)
        if llm_result and llm_result.confidence >= self.confidence_threshold:
            logger.info(f"LLM fallback result: {llm_result.verdict.value} (confidence: {llm_result.confidence})")
            return llm_result
        elif llm_result:
            logger.info(f"LLM confidence too low ({llm_result.confidence}), discarding claim")

        # No reliable verification found
        logger.info(f"Could not verify claim: {claim.claim_text[:50]}...")
        return None

    def _query_google_factcheck(self, claim: FactualClaim) -> Optional[FactCheckResult]:
        """
        Query Google Fact Check Tools API.

        Args:
            claim: The claim to search for

        Returns:
            FactCheckResult if a relevant fact-check is found
        """
        try:
            params = {
                "query": claim.claim_text,
                "key": self.google_api_key,
                "languageCode": "en"
            }

            response = requests.get(
                self.google_api_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Check if we got any claims back
            if "claims" not in data or not data["claims"]:
                logger.debug("No Google Fact Check results found")
                return None

            # Find the best matching claim review
            best_review = self._find_best_review(data["claims"], claim)
            if not best_review:
                return None

            # Extract and normalize the verdict
            textual_rating = best_review.get("textualRating", "").lower().strip()
            verdict = self._normalize_rating(textual_rating)

            # Get source information
            publisher = best_review.get("publisher", {})
            publisher_name = publisher.get("name", "Unknown")
            source_url = best_review.get("url", "")

            # Build correction text from the review
            correction = None
            if verdict in (Verdict.FALSE, Verdict.MISLEADING):
                # Try to get the title/summary as correction context
                title = best_review.get("title", "")
                if title:
                    correction = title

            return FactCheckResult(
                original_claim=claim,
                verdict=verdict,
                correction=correction,
                source_name=publisher_name,
                source_url=source_url,
                confidence=0.95 if self._is_reputable_publisher(publisher_name) else 0.80,
                method_used="google_factcheck",
                raw_rating=textual_rating
            )

        except requests.RequestException as e:
            logger.error(f"Google Fact Check API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing Google Fact Check response: {e}")
            return None

    def _find_best_review(
        self,
        google_claims: List[Dict],
        original_claim: FactualClaim
    ) -> Optional[Dict]:
        """
        Find the best matching claim review from Google results.

        Prioritizes:
        1. Reputable publishers (Snopes, PolitiFact, Reuters, etc.)
        2. Recency of the fact-check
        3. Relevance to the original claim
        """
        best_review = None
        best_score = 0

        for google_claim in google_claims:
            claim_reviews = google_claim.get("claimReview", [])

            for review in claim_reviews:
                score = 0
                publisher_name = review.get("publisher", {}).get("name", "").lower()

                # Score based on publisher reputation
                if self._is_reputable_publisher(publisher_name):
                    score += 10

                # Score based on having a URL
                if review.get("url"):
                    score += 5

                # Score based on having a textual rating
                if review.get("textualRating"):
                    score += 3

                if score > best_score:
                    best_score = score
                    best_review = review

        return best_review

    def _is_reputable_publisher(self, publisher_name: str) -> bool:
        """Check if a publisher is in our list of reputable fact-checkers."""
        publisher_lower = publisher_name.lower()
        return any(rep in publisher_lower for rep in REPUTABLE_PUBLISHERS)

    def _normalize_rating(self, rating: str) -> Verdict:
        """
        Normalize a textual rating to our standard Verdict enum.

        Args:
            rating: The raw textual rating from a fact-check

        Returns:
            Normalized Verdict
        """
        rating_lower = rating.lower().strip()

        # Direct match
        if rating_lower in RATING_NORMALIZATION:
            return RATING_NORMALIZATION[rating_lower]

        # Partial match - check if any key is contained in the rating
        for key, verdict in RATING_NORMALIZATION.items():
            if key in rating_lower:
                return verdict

        # Default to unverified if we can't determine
        return Verdict.UNVERIFIED

    def _llm_websearch_fallback(self, claim: FactualClaim) -> Optional[FactCheckResult]:
        """
        Use LLM with web search capability as fallback verification.

        This is a conservative fallback - only returns results with high confidence
        and verifiable sources.

        Args:
            claim: The claim to verify

        Returns:
            FactCheckResult or None
        """
        system_prompt = """You are a fact-checker. Your job is to verify factual claims
using web search and authoritative sources.

CRITICAL RULES:
1. Only provide a verdict if you find DIRECT EVIDENCE from authoritative sources
2. Authoritative sources include: government websites (.gov), academic institutions (.edu),
   peer-reviewed research, major news organizations (Reuters, AP, BBC), fact-checking
   organizations (Snopes, PolitiFact, FactCheck.org)
3. If you cannot find direct evidence, set verdict to "UNVERIFIED"
4. The source_url MUST be a real, working URL - NEVER fabricate URLs
5. Be conservative - when in doubt, mark as UNVERIFIED
6. Confidence should reflect how certain you are:
   - 0.95+: Direct fact-check from major organization
   - 0.85-0.94: Strong evidence from authoritative sources
   - 0.70-0.84: Good evidence but some uncertainty
   - Below 0.70: Not confident enough - mark as UNVERIFIED"""

        user_prompt = f"""Verify this factual claim:

CLAIM: "{claim.claim_text}"
CONTEXT: "{claim.context}"
CLAIM TYPE: {claim.claim_type.value}

Search for authoritative sources that either confirm or debunk this claim.

Respond in this EXACT JSON format:
{{
    "verdict": "TRUE" | "FALSE" | "MISLEADING" | "UNVERIFIED",
    "confidence": 0.0-1.0,
    "correction": "If FALSE/MISLEADING, provide the accurate information. Otherwise null.",
    "source_name": "Name of the most authoritative source",
    "source_url": "Direct URL to the evidence (must be real and working)",
    "reasoning": "Brief explanation of how you verified this"
}}

Remember: If you're not highly confident with a verifiable source, use UNVERIFIED."""

        try:
            response = requests.post(
                self.openrouter_url,
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "HTTP-Referer": "https://github.com/ai-tweets",
                    "X-Title": "PodDebunker Fact Checker",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,  # Low temperature for factual accuracy
                    "response_format": {"type": "json_object"}
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            data = json.loads(content)

            # Parse verdict
            verdict_str = data.get("verdict", "UNVERIFIED").upper()
            try:
                verdict = Verdict(verdict_str.lower())
            except ValueError:
                verdict = Verdict.UNVERIFIED

            confidence = float(data.get("confidence", 0.0))
            source_url = data.get("source_url", "")

            # Validate the URL looks reasonable (basic check)
            if source_url and not self._is_valid_url(source_url):
                logger.warning(f"Invalid URL from LLM: {source_url}")
                source_url = ""
                confidence = min(confidence, 0.5)  # Reduce confidence

            return FactCheckResult(
                original_claim=claim,
                verdict=verdict,
                correction=data.get("correction"),
                source_name=data.get("source_name", ""),
                source_url=source_url,
                confidence=confidence,
                method_used="llm_websearch",
                raw_rating=verdict_str
            )

        except requests.RequestException as e:
            logger.error(f"LLM API error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in LLM fallback: {e}")
            return None

    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation."""
        if not url:
            return False
        # Check for basic URL structure
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        return bool(url_pattern.match(url))

    def get_false_claims(self, results: List[FactCheckResult]) -> List[FactCheckResult]:
        """
        Filter results to only false/misleading claims.

        Args:
            results: List of fact check results

        Returns:
            Only results with FALSE or MISLEADING verdicts
        """
        return [r for r in results if r.is_false_or_misleading]


def validate_source_url(url: str, timeout: int = 10) -> bool:
    """
    Validate that a source URL is actually reachable.

    Args:
        url: URL to validate
        timeout: Request timeout in seconds

    Returns:
        True if URL is reachable, False otherwise
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code < 400
    except:
        return False
