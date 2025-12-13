#!/usr/bin/env python3
"""
Twitter Handle Finder
Discovers Twitter handles for podcasts, hosts, and guests.

This module now integrates with the OSINT Handle Finder for advanced
handle discovery using multiple sources (Wikidata, Twitter API, web search, etc.)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import OSINT Handle Finder (optional dependency)
try:
    from .osint_handle_finder import (
        OSINTHandleFinder,
        HandleLookupContext,
        LookupStatus
    )
    OSINT_AVAILABLE = True
except ImportError:
    OSINT_AVAILABLE = False
    logger.debug("OSINT Handle Finder not available, using basic lookup")


@dataclass
class PodcastHandles:
    """Twitter handles for a podcast and its participants."""
    podcast_handle: Optional[str] = None
    host_handles: List[str] = None

    def __post_init__(self):
        if self.host_handles is None:
            self.host_handles = []


class TwitterHandleFinder:
    """
    Finds and manages Twitter handles for podcasts and their participants.

    Features:
    - OSINT-based handle discovery (when available)
    - Manual mapping database for known podcasts/hosts
    - Guest name extraction from episode titles/descriptions
    - Multi-source confidence scoring
    - Human review queue for uncertain matches
    - Fallback to plain text names if no handle found
    """

    def __init__(self, handle_config: Dict[str, Dict] = None, use_osint: bool = True):
        """
        Initialize handle finder with configuration.

        Args:
            handle_config: Dict mapping podcast names to handle info:
                {
                    "Podcast Name": {
                        "podcast_handle": "@handle",
                        "host_handles": ["@host1", "@host2"]
                    }
                }
            use_osint: Whether to use OSINT Handle Finder for discovery
        """
        self.handle_config = handle_config or {}
        self.use_osint = use_osint and OSINT_AVAILABLE

        # Initialize OSINT finder if available
        self._osint_finder = None
        if self.use_osint:
            try:
                self._osint_finder = OSINTHandleFinder()
                logger.info("OSINT Handle Finder enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize OSINT finder: {e}")
                self._osint_finder = None

        # Known guest handles (fast lookup cache for common podcast guests)
        self.known_guests = {
            # Tech/Startup personalities
            "Andrew Huberman": "@AndrewHuberman",
            "Lex Fridman": "@lexfridman",
            "Tim Ferriss": "@tferriss",
            "Joe Rogan": "@joerogan",
            "Sam Harris": "@SamHarrisOrg",
            "Naval Ravikant": "@naval",
            "Chamath Palihapitiya": "@chamath",
            "Jason Calacanis": "@Jason",
            "David Sacks": "@DavidSacks",
            "David Friedberg": "@friedberg",
            "Elon Musk": "@elonmusk",
            "Mark Zuckerberg": "@finkd",
            "Jeff Bezos": "@JeffBezos",
            "Marc Andreessen": "@pmarca",
            "Ben Horowitz": "@bhorowitz",
            "Peter Thiel": "@peterthiel",
            "Reid Hoffman": "@reidhoffman",
            "Sam Altman": "@sama",
            "Satya Nadella": "@satloganadella",
            "Sundar Pichai": "@sundarpichai",
            # Entertainers/Celebrities (Diary of CEO guests)
            "Kevin Hart": "@KevinHart4real",
            "Steven Bartlett": "@StevenBartlett",
            "Gary Vaynerchuk": "@garyvee",
            "Tony Robbins": "@TonyRobbins",
            "Simon Sinek": "@simonsinek",
            "Matthew McConaughey": "@McConaughey",
            "Will Smith": "@willsmith",
            "Dwayne Johnson": "@TheRock",
            "Ryan Reynolds": "@VancityReynolds",
            "Chris Williamson": "@ChrisWillx",
            # Experts/Scientists
            "Tristan Harris": "@tristanharris",
            "David Spiegel": "@davidspiegel",
            "Peter Attia": "@PeterAttiaMD",
            "Rhonda Patrick": "@foundmyfitness",
            "Jordan Peterson": "@jordanbpeterson",
            "James Clear": "@JamesClear",
            "Cal Newport": "@calnewport",
            # Business leaders
            "Brian Chesky": "@bchesky",
            "Daniel Ek": "@eldsjal",
            "Jensen Huang": "@nvidia",
            "Dario Amodei": "@DarioAmodei",
            # Podcast guests from pending queue
            "Pavel Durov": "@durov",
            "Steven Pressfield": "@SPressfield",
            "Janna Levin": "@jannalevin",
            "Bret Contreras": "@bretcontreras1",
            "Dr. Bret Contreras": "@bretcontreras1",
            "Julia Shaw": "@drjuliashaw",
            "Dave Plummer": "@davepl1968",
            "Chris Dixon": "@cdixon",
            # Additional common podcast guests
            "Alex Hormozi": "@AlexHormozi",
            "Sahil Bloom": "@SahilBloom",
            "Shaan Puri": "@ShaanVP",
            "Patrick O'Shaughnessy": "@patrick_oshag",
            "Tyler Cowen": "@tylercowen",
            "Balaji Srinivasan": "@balaborz",
            "Ryan Holiday": "@RyanHoliday",
            "Tim Urban": "@waitbutwhy",
            "Derek Sivers": "@sivers",
            "Tim Denning": "@TimDenning",
            "Brene Brown": "@BreneBrown",
            "Adam Grant": "@AdamMGrant",
            "Malcolm Gladwell": "@Gladwell",
            "Seth Godin": "@ThisIsSethsBlog",
            "Guy Kawasaki": "@GuyKawasaki",
            "Nassim Taleb": "@nntaleb",
            "Michael Saylor": "@saylor",
            "Ray Dalio": "@RayDalio",
            "Jocko Willink": "@jaborz",
            "David Goggins": "@davidgoggins",
            "Mel Robbins": "@melrobbins",
            "Jay Shetty": "@JayShetty",
            "Lewis Howes": "@LewisHowes",
            "Marie Forleo": "@marieforleo",
            "Arianna Huffington": "@araborzhuff",
            "Oprah Winfrey": "@Oprah",
            "Michelle Obama": "@MichelleObama",
            "Barack Obama": "@BarackObama",
            "Bill Gates": "@BillGates",
            "Warren Buffett": "@WarrenBuffett",
            "Kara Swisher": "@karaswisher",
            "Scott Galloway": "@profgalloway",
            "Ezra Klein": "@ezraklein",
            "Ben Shapiro": "@benshapiro",
            "Dave Rubin": "@RubinReport",
            "Russell Brand": "@rustyrockets",
            "Russ": "@russdiemon",
            "Ed Mylett": "@EdMylett",
            "Grant Cardone": "@GrantCardone",
            "Dr. Mark Hyman": "@drmarkhyman",
            "Dr. Gabor Mate": "@DrGaborMate",
            "Dr. Andrew Weil": "@DrWeil",
            "Dr. Michael Greger": "@nutrition_facts",
            "Rich Roll": "@richroll",
            "Mat Fraser": "@mathewfras",
            "Alex Cooper": "@alexandracooper",
            "Emma Chamberlain": "@emmachamberlain",
            "Mr. Beast": "@MrBeast",
            "Logan Paul": "@LoganPaul",
            "Jake Paul": "@jakepaul",
            "KSI": "@KSI",
            "Hasan Minhaj": "@hasanminhaj",
            "Trevor Noah": "@Trevornoah",
            "John Oliver": "@iamjohnoliver",
        }

        logger.info(
            f"TwitterHandleFinder initialized with {len(self.handle_config)} podcasts "
            f"(OSINT: {'enabled' if self._osint_finder else 'disabled'})"
        )

    def get_podcast_handles(self, podcast_name: str) -> PodcastHandles:
        """
        Get handles for a podcast and its hosts.

        Args:
            podcast_name: Name of the podcast

        Returns:
            PodcastHandles object with podcast and host handles
        """
        config = self.handle_config.get(podcast_name, {})

        return PodcastHandles(
            podcast_handle=config.get('podcast_handle'),
            host_handles=config.get('host_handles', [])
        )

    def extract_guest_names(self, episode_title: str, episode_description: str = "") -> List[str]:
        """
        Extract guest names from episode title and description.

        Common patterns:
        - "Episode Title | Guest Name"
        - "Episode Title with Guest Name"
        - "Guest Name: Episode Title"
        - "Episode Title ft. Guest Name"

        Args:
            episode_title: Episode title
            episode_description: Episode description (optional)

        Returns:
            List of extracted guest names
        """
        guests = []

        # Pattern 1: "Title | Guest Name"
        if '|' in episode_title:
            parts = episode_title.split('|')
            if len(parts) >= 2:
                potential_guest = parts[1].strip()
                # Remove common suffixes like "PhD", "MD", "Dr."
                potential_guest = re.sub(r'\s+(PhD|MD|Dr\.|M\.D\.|Ph\.D\.).*$', '', potential_guest)
                if potential_guest and len(potential_guest) > 3:
                    guests.append(potential_guest)

        # Pattern 2: "with Guest Name"
        with_match = re.search(r'\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', episode_title)
        if with_match:
            guests.append(with_match.group(1).strip())

        # Pattern 3: "ft. Guest Name" or "feat. Guest Name"
        feat_match = re.search(r'\b(?:ft\.|feat\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', episode_title)
        if feat_match:
            guests.append(feat_match.group(1).strip())

        # Pattern 4: "Guest Name:" at start (Diary of CEO style)
        # Exclude non-guest prefixes like "Most Replayed Moment", "Essentials", etc.
        non_guest_prefixes = [
            'Most Replayed Moment', 'Essentials', 'Best Of', 'Bonus', 'Trailer',
            'Highlight', 'Preview', 'Announcement', 'Update', 'Special',
            'AI Expert', 'Insulin Doctor', 'Female Hormone Health'  # Topic prefixes
        ]
        colon_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+):', episode_title)
        if colon_match:
            potential_guest = colon_match.group(1).strip()
            # Only add if not a non-guest prefix
            if potential_guest not in non_guest_prefixes:
                guests.append(potential_guest)

        # Pattern 5: "#NNN – Guest Name" (Lex Fridman style)
        lex_match = re.match(r'^#\d+\s*[–-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', episode_title)
        if lex_match:
            potential_guest = lex_match.group(1).strip()
            # Remove topic suffix if present (e.g., "Name: Topic")
            if ':' in potential_guest:
                potential_guest = potential_guest.split(':')[0].strip()
            guests.append(potential_guest)

        # Pattern 6: "E123: Guest Name -" or "Ep. 123: Guest Name"
        ep_num_match = re.match(r'^(?:E|Ep\.?)\s*\d+[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', episode_title)
        if ep_num_match:
            potential_guest = ep_num_match.group(1).strip()
            guests.append(potential_guest)

        # Pattern 7: "Guest Name - Topic" (dash separator)
        dash_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[-–—]\s*', episode_title)
        if dash_match and '|' not in episode_title:  # Don't conflict with pattern 1
            potential_guest = dash_match.group(1).strip()
            if potential_guest not in non_guest_prefixes:
                guests.append(potential_guest)

        # Pattern 8: "Interview with Guest Name" or "Conversation with Guest Name"
        interview_match = re.search(r'(?:Interview|Conversation|Talk|Chat)\s+with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', episode_title, re.IGNORECASE)
        if interview_match:
            guests.append(interview_match.group(1).strip())

        # Pattern 9: Extract from description - "In this episode, [Name] discusses..."
        if episode_description:
            desc_patterns = [
                r'(?:In this episode|Today\'s guest|Our guest),?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'(?:speaks with|interviews|welcomes)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:is|joins|shares|discusses|reveals)',
            ]
            for pattern in desc_patterns:
                desc_match = re.search(pattern, episode_description[:500])  # Only check first 500 chars
                if desc_match:
                    guests.append(desc_match.group(1).strip())
                    break  # Only add one from description

        # Deduplicate and clean
        guests = list(set(guests))
        guests = [g for g in guests if self._is_likely_person_name(g)]

        logger.debug(f"Extracted {len(guests)} guest names from: {episode_title}")
        return guests

    def _is_likely_person_name(self, name: str) -> bool:
        """
        Check if a string is likely a person's name.

        Args:
            name: Potential name string

        Returns:
            True if likely a person name
        """
        # Must have at least first and last name
        parts = name.split()
        if len(parts) < 2:
            return False

        # Each part should start with capital letter
        if not all(part[0].isupper() for part in parts if part):
            return False

        # Shouldn't be too long (likely a phrase, not a name)
        if len(parts) > 4:
            return False

        # Exclude common non-name patterns
        exclude_patterns = [
            'Episode', 'Podcast', 'Show', 'Interview', 'Talk',
            'Part', 'Series', 'Season', 'Special', 'Live'
        ]
        if any(pattern in name for pattern in exclude_patterns):
            return False

        return True

    def find_handles(
        self,
        names: List[str],
        podcast_name: str = None,
        episode_title: str = None,
        episode_description: str = None
    ) -> List[str]:
        """
        Find Twitter handles for a list of names.

        Uses OSINT Handle Finder when available, falls back to
        known guests database, then plain names.

        Args:
            names: List of person names
            podcast_name: Optional podcast name for context
            episode_title: Optional episode title for context
            episode_description: Optional description for context

        Returns:
            List of Twitter handles (or plain names if handle not found)
        """
        handles = []

        for name in names:
            handle = self._find_single_handle(
                name, podcast_name, episode_title, episode_description
            )
            handles.append(handle)

        return handles

    def _find_single_handle(
        self,
        name: str,
        podcast_name: str = None,
        episode_title: str = None,
        episode_description: str = None
    ) -> str:
        """
        Find a Twitter handle for a single name.

        Args:
            name: Person name
            podcast_name: Optional podcast name
            episode_title: Optional episode title
            episode_description: Optional description

        Returns:
            Twitter handle with @ prefix, or plain name if not found
        """
        # 1. Check known guests first (fast path)
        if name in self.known_guests:
            handle = self.known_guests[name]
            logger.debug(f"Found handle in known guests: {name} -> {handle}")
            return handle

        # 2. Try OSINT finder if available
        if self._osint_finder:
            try:
                # Build context
                context = None
                if OSINT_AVAILABLE and (podcast_name or episode_title or episode_description):
                    context = HandleLookupContext(
                        podcast_name=podcast_name,
                        episode_title=episode_title,
                        episode_description=episode_description
                    )

                result = self._osint_finder.find_handle(name, context)

                if result.handle and result.status in [
                    LookupStatus.VERIFIED,
                    LookupStatus.MANUAL_APPROVED,
                    LookupStatus.CACHED
                ]:
                    handle = f"@{result.handle}"
                    logger.info(
                        f"OSINT found handle for {name}: {handle} "
                        f"(confidence: {result.confidence:.0%})"
                    )
                    return handle

                elif result.status == LookupStatus.PENDING_REVIEW:
                    # Use the handle but note it needs review
                    if result.handle:
                        handle = f"@{result.handle}"
                        logger.info(
                            f"OSINT found handle for {name}: {handle} "
                            f"(pending review, confidence: {result.confidence:.0%})"
                        )
                        return handle

            except Exception as e:
                logger.warning(f"OSINT lookup failed for {name}: {e}")

        # 3. Fallback: use plain name
        logger.debug(f"No handle found for {name}, using plain name")
        return name

    def find_handle_with_confidence(
        self,
        name: str,
        podcast_name: str = None,
        episode_title: str = None
    ) -> Tuple[str, float, str]:
        """
        Find a handle with confidence score and status.

        Args:
            name: Person name
            podcast_name: Optional podcast name
            episode_title: Optional episode title

        Returns:
            Tuple of (handle_or_name, confidence, status)
        """
        # Check known guests
        if name in self.known_guests:
            return (self.known_guests[name], 1.0, "known_guest")

        # Try OSINT
        if self._osint_finder:
            try:
                context = None
                if OSINT_AVAILABLE and podcast_name:
                    context = HandleLookupContext(
                        podcast_name=podcast_name,
                        episode_title=episode_title
                    )

                result = self._osint_finder.find_handle(name, context)

                if result.handle:
                    return (
                        f"@{result.handle}",
                        result.confidence,
                        result.status.value
                    )

            except Exception as e:
                logger.warning(f"OSINT lookup failed for {name}: {e}")

        return (name, 0.0, "not_found")

    def get_all_handles_for_episode(
        self,
        podcast_name: str,
        episode_title: str,
        episode_description: str = ""
    ) -> Tuple[Optional[str], List[str], List[str]]:
        """
        Get all handles for an episode: podcast, hosts, and guests.

        Args:
            podcast_name: Name of the podcast
            episode_title: Episode title
            episode_description: Episode description (optional)

        Returns:
            Tuple of (podcast_handle, host_handles, guest_handles)
        """
        # Get podcast and host handles
        podcast_handles = self.get_podcast_handles(podcast_name)

        # Extract and find guest handles with context
        guest_names = self.extract_guest_names(episode_title, episode_description)
        guest_handles = self.find_handles(
            guest_names,
            podcast_name=podcast_name,
            episode_title=episode_title,
            episode_description=episode_description
        )

        logger.info(
            f"Handles for {podcast_name}: "
            f"Podcast={podcast_handles.podcast_handle}, "
            f"Hosts={len(podcast_handles.host_handles)}, "
            f"Guests={len(guest_handles)}"
        )

        return (
            podcast_handles.podcast_handle,
            podcast_handles.host_handles,
            guest_handles
        )

    def add_known_guest(self, name: str, handle: str):
        """
        Add a known guest to the database.

        Args:
            name: Full name of the guest
            handle: Twitter handle (with @)
        """
        if not handle.startswith('@'):
            handle = f'@{handle}'

        self.known_guests[name] = handle
        logger.info(f"Added known guest: {name} -> {handle}")
