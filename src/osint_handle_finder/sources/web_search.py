"""
Web search source for finding Twitter handles.

Uses DuckDuckGo HTML scraping (no API key required) to search for
Twitter/X.com profiles associated with a person's name.
"""

import requests
import re
import logging
import time
import urllib.parse
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup

from .base import HandleSource
from ..models import HandleCandidate, HandleLookupContext

logger = logging.getLogger(__name__)


class DuckDuckGoSource(HandleSource):
    """
    DuckDuckGo web search source for Twitter handle discovery.

    Searches for "[name] twitter" and extracts handles from results.
    Uses HTML scraping - no API key required.
    """

    priority = 20  # Medium priority
    base_confidence = 0.65  # Lower confidence - search results can be noisy
    requires_auth = False
    source_name = "duckduckgo"

    # DuckDuckGo HTML search URL
    SEARCH_URL = "https://html.duckduckgo.com/html/"
    TIMEOUT = 15  # seconds

    # Rate limiting - be respectful
    MIN_REQUEST_INTERVAL = 2.0  # seconds between requests
    _last_request_time = 0.0

    # Twitter/X URL patterns
    TWITTER_URL_PATTERNS = [
        r'(?:twitter\.com|x\.com)/([a-zA-Z0-9_]{1,15})(?:/|$|\?)',
    ]

    def is_available(self) -> bool:
        """Check if DuckDuckGo is reachable."""
        try:
            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": "test"},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def lookup(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[HandleCandidate]:
        """
        Search for Twitter handles via DuckDuckGo.

        Args:
            name: Person name to search for
            context: Optional context for better results

        Returns:
            List of handle candidates
        """
        candidates = []

        # Build search queries
        queries = self._build_search_queries(name, context)

        for query in queries[:3]:  # Limit queries to avoid being blocked
            self._wait_for_rate_limit()

            try:
                results = self._search(query)
                handles = self._extract_handles_from_results(results)

                for handle, snippet, url in handles:
                    confidence = self._calculate_confidence(name, handle, snippet, context)
                    if confidence > 0.25:  # Minimum threshold
                        evidence = {
                            "search_query": query,
                            "result_snippet": snippet[:300] if snippet else None,
                            "source_url": url,
                            "match_type": "web_search"
                        }

                        candidates.append(self._create_candidate(
                            handle=handle,
                            confidence=confidence,
                            evidence=evidence
                        ))

            except Exception as e:
                logger.error(f"DuckDuckGo search error for '{query}': {e}")
                continue

        # Deduplicate by handle, keeping highest confidence
        unique_candidates = self._deduplicate_candidates(candidates)
        return unique_candidates

    def _build_search_queries(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[str]:
        """
        Build search queries for finding Twitter handles.

        Args:
            name: Person name
            context: Optional context

        Returns:
            List of search queries to try
        """
        queries = [
            f'"{name}" twitter',
            f'"{name}" site:twitter.com',
            f'"{name}" site:x.com',
        ]

        # Add context-enhanced queries
        if context:
            if context.podcast_name:
                queries.insert(0, f'"{name}" "{context.podcast_name}" twitter')

            if context.known_profession:
                queries.append(f'"{name}" {context.known_profession} twitter')

            if context.known_company:
                queries.append(f'"{name}" {context.known_company} twitter')

        return queries[:5]  # Limit total queries

    def _search(self, query: str) -> List[Tuple[str, str, str]]:
        """
        Perform a DuckDuckGo HTML search.

        Args:
            query: Search query

        Returns:
            List of (title, snippet, url) tuples
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        response = requests.post(
            self.SEARCH_URL,
            data={"q": query, "b": ""},
            headers=headers,
            timeout=self.TIMEOUT
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []

        # Parse DuckDuckGo HTML results
        for result in soup.select('.result'):
            title_elem = result.select_one('.result__title')
            snippet_elem = result.select_one('.result__snippet')
            link_elem = result.select_one('.result__url')

            title = title_elem.get_text(strip=True) if title_elem else ""
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            # Get the actual URL
            url = ""
            if link_elem:
                url = link_elem.get_text(strip=True)
            elif title_elem:
                link = title_elem.select_one('a')
                if link and link.get('href'):
                    url = link.get('href')

            results.append((title, snippet, url))

        return results[:15]  # Limit results

    def _extract_handles_from_results(
        self,
        results: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract Twitter handles from search results.

        Args:
            results: List of (title, snippet, url) tuples

        Returns:
            List of (handle, snippet, url) tuples
        """
        handles = []

        for title, snippet, url in results:
            # Check URL for Twitter/X patterns
            for pattern in self.TWITTER_URL_PATTERNS:
                for text in [url, title]:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        handle = match.lower()
                        # Skip invalid handles
                        if self._is_valid_handle(handle):
                            handles.append((handle, snippet, url))

            # Also look for @mentions in snippets
            at_mentions = re.findall(r'@([a-zA-Z0-9_]{1,15})\b', snippet)
            for mention in at_mentions:
                if self._is_valid_handle(mention):
                    handles.append((mention.lower(), snippet, url))

        return handles

    def _calculate_confidence(
        self,
        target_name: str,
        handle: str,
        snippet: str,
        context: Optional[HandleLookupContext] = None
    ) -> float:
        """
        Calculate confidence for a handle found via search.

        Args:
            target_name: Name we're searching for
            handle: Found handle
            snippet: Search result snippet
            context: Optional context

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.4  # Base confidence for search results

        target_lower = target_name.lower()
        snippet_lower = snippet.lower()

        # Name appears in snippet
        if target_lower in snippet_lower:
            confidence += 0.15

        # Handle matches name pattern
        target_parts = target_lower.split()
        if len(target_parts) >= 2:
            first = target_parts[0]
            last = target_parts[-1]

            # Handle contains name parts
            if first in handle and last in handle:
                confidence += 0.20
            elif first in handle or last in handle:
                confidence += 0.10

            # Name in reverse order (lastname_firstname)
            if last in handle and first in handle:
                confidence += 0.05

        # Snippet mentions Twitter/X profile
        if any(term in snippet_lower for term in ['twitter profile', 'x profile', 'on twitter', 'on x']):
            confidence += 0.10

        # Context matching
        if context:
            if context.podcast_name and context.podcast_name.lower() in snippet_lower:
                confidence += 0.15

            if context.known_profession and context.known_profession.lower() in snippet_lower:
                confidence += 0.10

            if context.known_company and context.known_company.lower() in snippet_lower:
                confidence += 0.10

        # Penalize generic handles
        generic_patterns = ['official', 'real', 'the', 'mr', 'ms', 'dr']
        if any(handle.startswith(p) for p in generic_patterns):
            confidence -= 0.05

        return min(1.0, max(0.0, confidence))

    def _deduplicate_candidates(
        self,
        candidates: List[HandleCandidate]
    ) -> List[HandleCandidate]:
        """
        Deduplicate candidates, keeping highest confidence for each handle.

        Args:
            candidates: List of candidates

        Returns:
            Deduplicated list
        """
        handle_map = {}
        for c in candidates:
            handle_lower = c.handle.lower()
            if handle_lower not in handle_map:
                handle_map[handle_lower] = c
            elif c.raw_confidence > handle_map[handle_lower].raw_confidence:
                handle_map[handle_lower] = c

        return list(handle_map.values())

    @staticmethod
    def _is_valid_handle(handle: str) -> bool:
        """
        Validate a Twitter handle.

        Args:
            handle: Handle to validate

        Returns:
            True if valid
        """
        if not handle or len(handle) < 1 or len(handle) > 15:
            return False

        # Must be alphanumeric or underscore
        if not all(c.isalnum() or c == '_' for c in handle):
            return False

        # Can't be all numbers
        if handle.isdigit():
            return False

        # Skip common non-person handles
        skip_handles = {
            'twitter', 'x', 'home', 'search', 'explore', 'notifications',
            'messages', 'settings', 'help', 'about', 'privacy', 'tos',
            'status', 'intent', 'share', 'hashtag', 'i', 'login'
        }
        if handle.lower() in skip_handles:
            return False

        return True
