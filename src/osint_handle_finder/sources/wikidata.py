"""
Wikidata source for finding Twitter handles.

Uses the Wikidata SPARQL endpoint to find official Twitter handles
linked to notable people's Wikidata entries (P2002 property).

This is the most reliable source for notable people as these are
typically verified official links.
"""

import requests
import logging
import urllib.parse
from typing import List, Optional

from .base import HandleSource
from ..models import HandleCandidate, HandleLookupContext

logger = logging.getLogger(__name__)


class WikidataSource(HandleSource):
    """
    Wikidata SPARQL source for Twitter handles.

    Uses the P2002 (Twitter username) property from Wikidata.
    Highly reliable for notable people.
    """

    priority = 15  # High priority (lower = higher)
    base_confidence = 0.95  # Very high confidence for official links
    requires_auth = False
    source_name = "wikidata"

    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    TIMEOUT = 10  # seconds

    def is_available(self) -> bool:
        """Check if Wikidata API is reachable."""
        try:
            response = requests.get(
                self.SPARQL_ENDPOINT,
                params={"query": "ASK { ?s ?p ?o }", "format": "json"},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def lookup(
        self,
        name: str,
        context: Optional[HandleLookupContext] = None
    ) -> List[HandleCandidate]:
        """
        Look up Twitter handles from Wikidata.

        Args:
            name: Person name to search for
            context: Optional context (not heavily used for Wikidata)

        Returns:
            List of handle candidates
        """
        candidates = []

        try:
            # Try exact name match first
            results = self._query_by_name(name)

            if results:
                for result in results[:5]:  # Limit to top 5 matches
                    handle = result.get('twitter', '').strip()
                    if handle and self._is_valid_handle(handle):
                        # Calculate confidence boost based on available data
                        confidence = 1.0

                        # Boost if has Wikipedia article
                        if result.get('article'):
                            confidence += 0.05

                        # Boost if description matches context
                        description = result.get('description', '').lower()
                        if context and context.known_profession:
                            if context.known_profession.lower() in description:
                                confidence += 0.10

                        evidence = {
                            "wikidata_id": result.get('person', '').split('/')[-1] if result.get('person') else None,
                            "label": result.get('personLabel'),
                            "description": result.get('description'),
                            "wikipedia_article": result.get('article'),
                            "match_type": "exact_name"
                        }

                        candidates.append(self._create_candidate(
                            handle=handle,
                            confidence=confidence,
                            evidence=evidence
                        ))

            # If no results, try alternative name formats
            if not candidates:
                alt_results = self._query_by_alternative_names(name)
                for result in alt_results[:3]:
                    handle = result.get('twitter', '').strip()
                    if handle and self._is_valid_handle(handle):
                        evidence = {
                            "wikidata_id": result.get('person', '').split('/')[-1] if result.get('person') else None,
                            "label": result.get('personLabel'),
                            "description": result.get('description'),
                            "match_type": "alternative_name"
                        }
                        candidates.append(self._create_candidate(
                            handle=handle,
                            confidence=0.85,  # Lower confidence for alt name match
                            evidence=evidence
                        ))

        except Exception as e:
            logger.error(f"Wikidata lookup error for {name}: {e}")

        return candidates

    def _query_by_name(self, name: str) -> List[dict]:
        """
        Query Wikidata for a person by exact name.

        Args:
            name: Person name

        Returns:
            List of result dictionaries
        """
        # SPARQL query to find humans with Twitter accounts matching the name
        query = '''
        SELECT DISTINCT ?person ?personLabel ?twitter ?description ?article WHERE {
          ?person wdt:P31 wd:Q5;           # Instance of human
                  rdfs:label ?label;
                  wdt:P2002 ?twitter.       # Has Twitter username

          # Match the name (case-insensitive)
          FILTER(LCASE(?label) = LCASE("%s"@en) || LCASE(?label) = LCASE("%s"))

          # Get English label
          FILTER(LANG(?label) = "en" || LANG(?label) = "")

          # Optional description
          OPTIONAL {
            ?person schema:description ?description.
            FILTER(LANG(?description) = "en")
          }

          # Optional Wikipedia article
          OPTIONAL {
            ?article schema:about ?person;
                     schema:isPartOf <https://en.wikipedia.org/>.
          }

          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 10
        ''' % (name.replace('"', '\\"'), name.replace('"', '\\"'))

        return self._execute_query(query)

    def _query_by_alternative_names(self, name: str) -> List[dict]:
        """
        Query Wikidata using alternative name properties (aliases).

        Args:
            name: Person name

        Returns:
            List of result dictionaries
        """
        # Query using alternative names/aliases
        query = '''
        SELECT DISTINCT ?person ?personLabel ?twitter ?description WHERE {
          ?person wdt:P31 wd:Q5;
                  wdt:P2002 ?twitter;
                  skos:altLabel ?alias.

          FILTER(LCASE(?alias) = LCASE("%s"@en) || CONTAINS(LCASE(?alias), LCASE("%s")))
          FILTER(LANG(?alias) = "en" || LANG(?alias) = "")

          OPTIONAL {
            ?person schema:description ?description.
            FILTER(LANG(?description) = "en")
          }

          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        LIMIT 5
        ''' % (name.replace('"', '\\"'), name.replace('"', '\\"'))

        return self._execute_query(query)

    def _execute_query(self, query: str) -> List[dict]:
        """
        Execute a SPARQL query against Wikidata.

        Args:
            query: SPARQL query string

        Returns:
            List of result bindings as dictionaries
        """
        try:
            response = requests.get(
                self.SPARQL_ENDPOINT,
                params={
                    "query": query,
                    "format": "json"
                },
                headers={
                    "User-Agent": "PodcastTLDR/1.0 (Twitter Handle Finder)",
                    "Accept": "application/sparql-results+json"
                },
                timeout=self.TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                for binding in data.get('results', {}).get('bindings', []):
                    result = {}
                    for key, value in binding.items():
                        result[key] = value.get('value', '')
                    results.append(result)

                return results

            else:
                logger.warning(f"Wikidata query returned status {response.status_code}")
                return []

        except requests.Timeout:
            logger.warning("Wikidata query timed out")
            return []
        except Exception as e:
            logger.error(f"Wikidata query error: {e}")
            return []

    @staticmethod
    def _is_valid_handle(handle: str) -> bool:
        """
        Check if a handle looks valid.

        Args:
            handle: Twitter handle to validate

        Returns:
            True if handle appears valid
        """
        if not handle:
            return False

        # Remove @ if present
        handle = handle.lstrip('@')

        # Basic validation
        if len(handle) < 1 or len(handle) > 15:
            return False

        # Must be alphanumeric or underscore
        if not all(c.isalnum() or c == '_' for c in handle):
            return False

        # Can't be all numbers
        if handle.isdigit():
            return False

        # Reserved words
        reserved = {'twitter', 'admin', 'support', 'help', 'home', 'search'}
        if handle.lower() in reserved:
            return False

        return True
