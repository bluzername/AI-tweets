#!/usr/bin/env python3
"""
CLI interface for the OSINT Handle Finder.

Usage:
    python -m osint_handle_finder.cli lookup "Elon Musk"
    python -m osint_handle_finder.cli lookup "Naval Ravikant" --podcast "Tim Ferriss Show"
    python -m osint_handle_finder.cli review --list
    python -m osint_handle_finder.cli review --approve <review_id> <handle>
    python -m osint_handle_finder.cli stats
"""

import argparse
import sys
import json
import logging
from typing import Optional

from .handle_finder import OSINTHandleFinder
from .models import HandleLookupContext, LookupStatus


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def cmd_lookup(args):
    """Handle the 'lookup' command."""
    finder = OSINTHandleFinder()

    # Build context if provided
    context = None
    if args.podcast or args.profession or args.company:
        context = HandleLookupContext(
            podcast_name=args.podcast,
            known_profession=args.profession,
            known_company=args.company,
            episode_title=args.episode
        )

    # Perform lookup
    result = finder.find_handle(
        name=args.name,
        context=context,
        force_refresh=args.refresh
    )

    # Output results
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_result(result)


def print_result(result):
    """Pretty print a lookup result."""
    status_emoji = {
        LookupStatus.VERIFIED: "[OK]",
        LookupStatus.PENDING_REVIEW: "[?]",
        LookupStatus.NOT_FOUND: "[X]",
        LookupStatus.MANUAL_APPROVED: "[OK]",
        LookupStatus.MANUAL_REJECTED: "[X]",
        LookupStatus.CACHED: "[C]"
    }

    emoji = status_emoji.get(result.status, "")
    print(f"\n{emoji} Handle Lookup: {result.name}")
    print("-" * 50)

    if result.handle:
        confidence_bar = "=" * int(result.confidence * 20)
        print(f"Handle:     @{result.handle}")
        print(f"Confidence: [{confidence_bar:<20}] {result.confidence:.1%}")
        print(f"Status:     {result.status.value}")
        print(f"Sources:    {', '.join(result.sources_used)}")

        if result.review_id:
            print(f"Review ID:  {result.review_id}")
    else:
        print("No handle found")
        print(f"Status:     {result.status.value}")
        print(f"Sources searched: {', '.join(result.sources_used)}")

    print(f"Lookup time: {result.lookup_time_ms}ms")

    if result.candidates:
        print(f"\nTop candidates ({len(result.candidates)} total):")
        # Group by handle and show top 5
        seen = set()
        shown = 0
        for c in sorted(result.candidates, key=lambda x: x.raw_confidence, reverse=True):
            if c.handle not in seen and shown < 5:
                print(f"  - @{c.handle} ({c.source}: {c.raw_confidence:.2%})")
                seen.add(c.handle)
                shown += 1


def cmd_review(args):
    """Handle the 'review' command."""
    finder = OSINTHandleFinder()

    if args.list:
        reviews = finder.get_pending_reviews(limit=args.limit or 20)
        print(f"\nPending Reviews ({len(reviews)} items):")
        print("-" * 60)

        for item in reviews:
            top_handle = None
            top_score = 0
            for h, s in item.confidence_scores.items():
                if s > top_score:
                    top_handle = h
                    top_score = s

            print(f"\n[{item.review_id}] {item.name}")
            if top_handle:
                print(f"  Top candidate: @{top_handle} ({top_score:.1%})")
            print(f"  Candidates: {len(item.candidates)}")
            if item.context:
                if item.context.podcast_name:
                    print(f"  Podcast: {item.context.podcast_name}")
            print(f"  Created: {item.created_at}")

    elif args.approve:
        review_id = args.approve
        handle = args.handle

        if not handle:
            print("Error: --handle required with --approve")
            sys.exit(1)

        success = finder.approve_review(
            review_id=review_id,
            approved_handle=handle,
            reviewer="cli",
            notes=args.notes or ""
        )

        if success:
            print(f"[OK] Approved @{handle} for review {review_id}")
        else:
            print(f"[X] Failed to approve review {review_id}")
            sys.exit(1)

    elif args.reject:
        review_id = args.reject

        success = finder.reject_review(
            review_id=review_id,
            reviewer="cli",
            notes=args.notes or "Rejected via CLI"
        )

        if success:
            print(f"[OK] Rejected review {review_id}")
        else:
            print(f"[X] Failed to reject review {review_id}")
            sys.exit(1)

    elif args.detail:
        reviews = finder.get_pending_reviews(limit=100)
        for item in reviews:
            if item.review_id == args.detail:
                print(f"\nReview Detail: {item.review_id}")
                print("-" * 50)
                print(f"Name: {item.name}")
                print(f"Status: {item.status}")
                print(f"Created: {item.created_at}")

                if item.context:
                    print("\nContext:")
                    ctx = item.context
                    if ctx.podcast_name:
                        print(f"  Podcast: {ctx.podcast_name}")
                    if ctx.episode_title:
                        print(f"  Episode: {ctx.episode_title}")
                    if ctx.known_profession:
                        print(f"  Profession: {ctx.known_profession}")

                print("\nCandidates:")
                for c in sorted(item.candidates, key=lambda x: x.raw_confidence, reverse=True):
                    score = item.confidence_scores.get(c.handle, c.raw_confidence)
                    print(f"  @{c.handle}")
                    print(f"    Source: {c.source}")
                    print(f"    Confidence: {score:.1%}")
                    if c.evidence:
                        for k, v in list(c.evidence.items())[:3]:
                            print(f"    {k}: {str(v)[:50]}")
                return

        print(f"Review {args.detail} not found")


def cmd_stats(args):
    """Handle the 'stats' command."""
    finder = OSINTHandleFinder()
    stats = finder.get_stats()

    print("\nOSINT Handle Finder Statistics")
    print("=" * 40)
    print(f"Active sources:    {', '.join(stats['active_sources'])}")
    print(f"Cached handles:    {stats['cached_handles']}")
    print(f"Negative cache:    {stats['negative_cache']}")
    print(f"Pending reviews:   {stats['pending_reviews']}")
    print(f"Approved reviews:  {stats['approved_reviews']}")
    print(f"Rejected reviews:  {stats['rejected_reviews']}")
    print(f"Lookups (24h):     {stats['lookups_24h']}")
    print(f"Avg lookup time:   {stats['avg_lookup_ms_24h']}ms")
    print(f"Cache hits (24h):  {stats['cache_hits_24h']}")

    if args.json:
        print("\nFull stats:")
        print(json.dumps(stats, indent=2))


def cmd_batch(args):
    """Handle the 'batch' command for bulk lookups."""
    finder = OSINTHandleFinder()

    # Read names from file
    names = []
    with open(args.file, 'r') as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith('#'):
                names.append(name)

    print(f"Processing {len(names)} names...")

    results = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] Looking up: {name}")
        result = finder.find_handle(name)

        results.append({
            "name": name,
            "handle": result.handle,
            "confidence": result.confidence,
            "status": result.status.value
        })

        # Simple progress output
        if result.handle:
            print(f"  -> @{result.handle} ({result.confidence:.0%})")
        else:
            print(f"  -> Not found")

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nResults:")
        print(json.dumps(results, indent=2))


def cmd_cleanup(args):
    """Handle the 'cleanup' command."""
    finder = OSINTHandleFinder()
    removed = finder.cleanup()
    print(f"Cleaned up {removed} expired cache entries")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OSINT Handle Finder - Find Twitter handles using OSINT techniques"
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Lookup command
    lookup_parser = subparsers.add_parser('lookup', help='Look up a Twitter handle')
    lookup_parser.add_argument('name', help='Person name to look up')
    lookup_parser.add_argument('--podcast', '-p', help='Podcast name (context)')
    lookup_parser.add_argument('--profession', help='Known profession (context)')
    lookup_parser.add_argument('--company', help='Known company (context)')
    lookup_parser.add_argument('--episode', help='Episode title (context)')
    lookup_parser.add_argument('--refresh', '-r', action='store_true', help='Skip cache')
    lookup_parser.add_argument('--json', '-j', action='store_true', help='JSON output')

    # Review command
    review_parser = subparsers.add_parser('review', help='Manage review queue')
    review_parser.add_argument('--list', '-l', action='store_true', help='List pending reviews')
    review_parser.add_argument('--limit', type=int, help='Limit results')
    review_parser.add_argument('--approve', '-a', metavar='REVIEW_ID', help='Approve a review')
    review_parser.add_argument('--reject', metavar='REVIEW_ID', help='Reject a review')
    review_parser.add_argument('--handle', help='Handle to approve (use with --approve)')
    review_parser.add_argument('--notes', '-n', help='Review notes')
    review_parser.add_argument('--detail', '-d', metavar='REVIEW_ID', help='Show review details')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.add_argument('--json', '-j', action='store_true', help='Full JSON output')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Bulk lookup from file')
    batch_parser.add_argument('file', help='File with names (one per line)')
    batch_parser.add_argument('--output', '-o', help='Output file (JSON)')

    # Cleanup command
    subparsers.add_parser('cleanup', help='Clean up expired cache entries')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Dispatch to command handler
    if args.command == 'lookup':
        cmd_lookup(args)
    elif args.command == 'review':
        cmd_review(args)
    elif args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'batch':
        cmd_batch(args)
    elif args.command == 'cleanup':
        cmd_cleanup(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
