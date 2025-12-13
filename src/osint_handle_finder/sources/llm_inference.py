"""
LLM inference source for Twitter handle discovery.

Uses OpenAI/OpenRouter to intelligently infer likely Twitter handles
based on person name, profession, and context. Also validates
candidates from other sources.

This is a fallback source with lower confidence, but can help
when other sources fail.
"""

import os
import json
import logging
import re
from typing import List, Optional, Dict, Any

from .base import HandleSource
from ..models import HandleCandidate, HandleLookupContext

logger = logging.getLogger(__name__)


class LLMInferenceSource(HandleSource):
    """
    LLM-based handle inference and validation.

    Uses GPT/Claude to generate likely handle patterns and
    validate candidates against context.
    """

    priority = 50  # Low priority - use as fallback
    base_confidence = 0.45  # Lower confidence - inference-based
    requires_auth = True
    source_name = "llm_inference"

    TIMEOUT = 30  # seconds

    # System prompt for handle inference
    INFERENCE_PROMPT = """You are an expert at finding Twitter/X.com handles for notable people.
Given a person's name and context, suggest likely Twitter handles they might use.

Rules:
- Twitter handles are 1-15 characters, alphanumeric and underscores only
- Common patterns: firstname_lastname, firstnamelastname, flastname, first_last, therealname
- Consider the person's profession, company, and brand
- Notable people often have shorter, simpler handles
- Return ONLY handles without @ symbol
- Return 1-5 most likely handles, ranked by likelihood

Output format: JSON array of objects with "handle" and "reasoning" keys.
Example: [{"handle": "elonmusk", "reasoning": "Simple combination of first and last name, common for notable figures"}]

ONLY return valid JSON. No other text."""

    VALIDATION_PROMPT = """You are validating whether a Twitter handle likely belongs to a specific person.

Given:
- Person name: {name}
- Twitter handle: @{handle}
- Context: {context}
- Profile bio: {bio}

Evaluate if this handle likely belongs to this person.

Output format: JSON object with:
- "likely_match": boolean
- "confidence": float 0-1
- "reasoning": string explaining your assessment

Consider:
- Does the handle match the name pattern?
- Does the bio align with the person's known profession/context?
- Are there red flags (parody account, different person with similar name)?

ONLY return valid JSON."""

    def __init__(self):
        """Initialize LLM source."""
        super().__init__()
        # Try OpenRouter first, then OpenAI
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self._client = None

    def is_available(self) -> bool:
        """Check if LLM API is available."""
        return bool(self.openrouter_key or self.openai_key)

    def _get_completion(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Get a completion from the LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User query

        Returns:
            Response text or None
        """
        import requests

        if self.openrouter_key:
            return self._openrouter_completion(system_prompt, user_prompt)
        elif self.openai_key:
            return self._openai_completion(system_prompt, user_prompt)
        return None

    def _openrouter_completion(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Get completion from OpenRouter."""
        import requests

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/podcast-tldr",
                },
                json={
                    "model": "openai/gpt-4o-mini",  # Fast and cheap
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=self.TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content")
            else:
                logger.warning(f"OpenRouter returned {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            return None

    def _openai_completion(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Get completion from OpenAI."""
        import requests

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=self.TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content")
            else:
                logger.warning(f"OpenAI returned {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

    def lookup(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[HandleCandidate]:
        """
        Infer likely Twitter handles using LLM.

        Args:
            name: Person name
            context: Optional context

        Returns:
            List of inferred handle candidates
        """
        if not self.is_available():
            logger.debug("LLM API not configured")
            return []

        candidates = []

        # Build context string
        context_parts = [f"Person: {name}"]
        if context:
            if context.podcast_name:
                context_parts.append(f"Appeared on podcast: {context.podcast_name}")
            if context.known_profession:
                context_parts.append(f"Profession: {context.known_profession}")
            if context.known_company:
                context_parts.append(f"Company: {context.known_company}")
            if context.episode_title:
                context_parts.append(f"Episode: {context.episode_title}")

        user_prompt = "\n".join(context_parts)
        user_prompt += "\n\nSuggest likely Twitter handles for this person."

        try:
            response = self._get_completion(self.INFERENCE_PROMPT, user_prompt)
            if response:
                # Parse JSON from response
                suggestions = self._parse_json_response(response)

                for idx, suggestion in enumerate(suggestions[:5]):
                    handle = suggestion.get("handle", "").strip().lower()
                    reasoning = suggestion.get("reasoning", "")

                    if not self._is_valid_handle(handle):
                        continue

                    # Confidence decreases for later suggestions
                    position_penalty = idx * 0.1
                    confidence = 0.7 - position_penalty

                    evidence = {
                        "llm_reasoning": reasoning,
                        "inference_rank": idx + 1,
                        "match_type": "llm_inference"
                    }

                    candidates.append(self._create_candidate(
                        handle=handle,
                        confidence=confidence,
                        evidence=evidence
                    ))

        except Exception as e:
            logger.error(f"LLM inference error: {e}")

        return candidates

    def validate_candidate(
        self,
        name: str,
        handle: str,
        context: Optional[HandleLookupContext] = None,
        profile_bio: str = ""
    ) -> Dict[str, Any]:
        """
        Use LLM to validate if a handle likely belongs to a person.

        Args:
            name: Person name
            handle: Handle to validate
            context: Optional context
            profile_bio: Profile bio if available

        Returns:
            Validation result dict
        """
        if not self.is_available():
            return {"likely_match": None, "confidence": 0.5, "reasoning": "LLM not available"}

        # Build context string
        context_str = ""
        if context:
            parts = []
            if context.podcast_name:
                parts.append(f"Podcast: {context.podcast_name}")
            if context.known_profession:
                parts.append(f"Profession: {context.known_profession}")
            if context.known_company:
                parts.append(f"Company: {context.known_company}")
            context_str = "; ".join(parts) if parts else "No additional context"
        else:
            context_str = "No additional context"

        prompt = self.VALIDATION_PROMPT.format(
            name=name,
            handle=handle.lstrip('@'),
            context=context_str,
            bio=profile_bio[:300] if profile_bio else "Not available"
        )

        try:
            response = self._get_completion(
                "You validate Twitter handle ownership. Return ONLY valid JSON.",
                prompt
            )

            if response:
                result = self._parse_json_response(response)
                if isinstance(result, dict):
                    return {
                        "likely_match": result.get("likely_match", False),
                        "confidence": float(result.get("confidence", 0.5)),
                        "reasoning": result.get("reasoning", "")
                    }

        except Exception as e:
            logger.error(f"LLM validation error: {e}")

        return {"likely_match": None, "confidence": 0.5, "reasoning": "Validation failed"}

    def _parse_json_response(self, response: str) -> Any:
        """
        Parse JSON from LLM response, handling markdown code blocks.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON (list or dict)
        """
        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find array or object
        array_match = re.search(r'\[[\s\S]*\]', response)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        obj_match = re.search(r'\{[\s\S]*\}', response)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse LLM response as JSON: {response[:100]}...")
        return []

    @staticmethod
    def _is_valid_handle(handle: str) -> bool:
        """Validate Twitter handle format."""
        if not handle or len(handle) < 1 or len(handle) > 15:
            return False

        handle = handle.lstrip('@')

        if not all(c.isalnum() or c == '_' for c in handle):
            return False

        if handle.isdigit():
            return False

        return True
