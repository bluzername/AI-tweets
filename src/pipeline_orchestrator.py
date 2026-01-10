"""
Pipeline Orchestrator - Coordinates the unified podcast processing workflow.

This module implements the split workflow:
1. RSS Feed → Transcription → Claim Extraction (runs once)
2. Split into two parallel paths:
   - Path A: TLDR/Summary → Main Account
   - Path B: Fact-Check → PodDebunker Account

The fact-check path only publishes if sufficient false claims are found.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .config import Config, AccountType
from .claim_extractor import ClaimExtractor, ExtractedClaims
from .fact_checker import FactChecker, FactCheckResult
from .debunk_thread_generator import DebunkThreadGenerator, DebunkThreadConfig
from .thread_generator import ThreadGenerator, ThreadStyle, Tweet
from .x_publisher import MultiAccountPublisher
from .ai_analyzer import PodcastHighlight

logger = logging.getLogger(__name__)


@dataclass
class EpisodeData:
    """Data container for a podcast episode being processed."""
    podcast_name: str
    episode_title: str
    episode_number: Optional[str] = None
    transcription: str = ""
    podcast_handles: Dict[str, str] = field(default_factory=dict)
    artwork_url: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of processing an episode through the pipeline."""
    episode: EpisodeData
    claims: Optional[ExtractedClaims] = None

    # Summary pipeline results
    summary_thread: Optional[List[Tweet]] = None
    summary_publish_result: Optional[Dict[str, Any]] = None

    # Debunk pipeline results
    fact_check_results: Optional[List[FactCheckResult]] = None
    debunk_thread: Optional[List[Tweet]] = None
    debunk_publish_result: Optional[Dict[str, Any]] = None

    # Status
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "podcast_name": self.episode.podcast_name,
            "episode_title": self.episode.episode_title,
            "claims_extracted": {
                "insights": len(self.claims.insights) if self.claims else 0,
                "factual_claims": len(self.claims.factual_claims) if self.claims else 0
            },
            "summary": {
                "thread_length": len(self.summary_thread) if self.summary_thread else 0,
                "publish_status": self.summary_publish_result.get("status") if self.summary_publish_result else None
            },
            "debunk": {
                "false_claims_found": len([r for r in (self.fact_check_results or []) if r.is_false_or_misleading]),
                "thread_length": len(self.debunk_thread) if self.debunk_thread else 0,
                "publish_status": self.debunk_publish_result.get("status") if self.debunk_publish_result else None
            },
            "errors": self.errors
        }


class PipelineOrchestrator:
    """
    Orchestrates the unified podcast processing pipeline.

    Flow:
    1. Claim Extraction (shared) - Extract insights AND factual claims
    2. Summary Pipeline - Generate TLDR thread → Publish to main account
    3. Fact-Check Pipeline - Verify claims → Generate debunk thread → Publish to PodDebunker
    """

    def __init__(self, config: Config):
        """
        Initialize pipeline orchestrator.

        Args:
            config: Application configuration
        """
        self.config = config

        # Initialize claim extractor (uses OpenRouter/DeepSeek)
        if config.openrouter_api_key:
            self.claim_extractor = ClaimExtractor(
                openrouter_api_key=config.openrouter_api_key,
                checkworthy_threshold=config.factcheck.checkworthy_threshold
            )
        else:
            self.claim_extractor = None
            logger.warning("OpenRouter API key not set - claim extraction disabled")

        # Initialize fact checker
        if config.openrouter_api_key:
            self.fact_checker = FactChecker(
                google_api_key=config.google_factcheck_api_key,
                openrouter_api_key=config.openrouter_api_key,
                confidence_threshold=config.factcheck.llm_confidence_threshold,
                checkworthy_threshold=config.factcheck.checkworthy_threshold
            )
        else:
            self.fact_checker = None
            logger.warning("OpenRouter API key not set - fact checking disabled")

        # Initialize debunk thread generator
        self.debunk_generator = DebunkThreadGenerator(
            config=DebunkThreadConfig(
                min_false_claims=config.factcheck.min_false_claims,
                include_misleading=config.factcheck.include_misleading
            )
        )

        # Initialize summary thread generator (uses OpenAI or OpenRouter)
        api_key = config.openai_api_key or config.openrouter_api_key
        if api_key:
            self.thread_generator = ThreadGenerator(
                api_key=api_key,
                model=config.gpt_model
            )
        else:
            self.thread_generator = None
            logger.warning("No AI API key set - thread generation disabled")

        # Initialize publisher
        self.publisher = MultiAccountPublisher.from_config_accounts(config.accounts)

        logger.info("PipelineOrchestrator initialized")

    def process_episode(
        self,
        episode: EpisodeData,
        run_summary: bool = True,
        run_factcheck: bool = True,
        dry_run: bool = False
    ) -> PipelineResult:
        """
        Process a single episode through both pipelines.

        Args:
            episode: Episode data with transcription
            run_summary: Whether to run the summary pipeline
            run_factcheck: Whether to run the fact-check pipeline
            dry_run: If True, don't actually publish

        Returns:
            PipelineResult with outcomes from both pipelines
        """
        result = PipelineResult(episode=episode)

        logger.info(f"Processing: {episode.podcast_name} - {episode.episode_title}")

        # PHASE 1: Shared Claim Extraction
        if self.claim_extractor and episode.transcription:
            try:
                result.claims = self.claim_extractor.extract_all(
                    transcription=episode.transcription,
                    podcast_name=episode.podcast_name,
                    episode_title=episode.episode_title,
                    max_insights=self.config.max_highlights
                )
                logger.info(
                    f"Extracted {len(result.claims.insights)} insights, "
                    f"{len(result.claims.factual_claims)} factual claims"
                )
            except Exception as e:
                error_msg = f"Claim extraction failed: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # PHASE 2A: Summary Pipeline
        if run_summary and result.claims and result.claims.insights:
            try:
                result.summary_thread, result.summary_publish_result = self._run_summary_pipeline(
                    episode=episode,
                    insights=result.claims.insights,
                    dry_run=dry_run
                )
            except Exception as e:
                error_msg = f"Summary pipeline failed: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # PHASE 2B: Fact-Check Pipeline
        if run_factcheck and self.config.factcheck.enabled and result.claims and result.claims.factual_claims:
            try:
                result.fact_check_results, result.debunk_thread, result.debunk_publish_result = \
                    self._run_factcheck_pipeline(
                        episode=episode,
                        claims=result.claims.factual_claims,
                        dry_run=dry_run
                    )
            except Exception as e:
                error_msg = f"Fact-check pipeline failed: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        return result

    def _run_summary_pipeline(
        self,
        episode: EpisodeData,
        insights: List[PodcastHighlight],
        dry_run: bool
    ) -> tuple:
        """Run the summary/TLDR pipeline."""
        logger.info("Running summary pipeline...")

        # Generate thread
        thread = None
        if self.thread_generator:
            # Get style from main account config
            main_account = self.config.get_summary_account()
            style = ThreadStyle(
                tone=main_account.tone if main_account else "casual",
                emoji_usage=main_account.emoji_usage if main_account else "minimal",
                hashtag_style=main_account.hashtag_style if main_account else "minimal",
                cta_style="medium"
            )

            thread = self.thread_generator.generate_thread(
                highlights=insights,
                podcast_name=episode.podcast_name,
                episode_title=episode.episode_title,
                episode_number=episode.episode_number,
                style=style,
                target_audience=main_account.target_audience if main_account else "General audience",
                podcast_handles=episode.podcast_handles
            )
            logger.info(f"Generated summary thread with {len(thread)} tweets")

        # Publish to main account
        publish_result = None
        if thread:
            publish_result = self.publisher.publish_by_type(
                account_type=AccountType.SUMMARY,
                thread=thread,
                podcast_name=episode.podcast_name,
                episode_title=episode.episode_title,
                dry_run=dry_run or self.config.dry_run
            )
            logger.info(f"Summary publish result: {publish_result.get('status')}")

        return thread, publish_result

    def _run_factcheck_pipeline(
        self,
        episode: EpisodeData,
        claims: list,
        dry_run: bool
    ) -> tuple:
        """Run the fact-check pipeline."""
        logger.info(f"Running fact-check pipeline on {len(claims)} claims...")

        # Verify claims
        fact_check_results = []
        if self.fact_checker:
            fact_check_results = self.fact_checker.check_claims(claims)
            false_claims = self.fact_checker.get_false_claims(fact_check_results)
            logger.info(f"Found {len(false_claims)} false/misleading claims out of {len(fact_check_results)} verified")

        # Generate debunk thread (only if enough false claims)
        thread = None
        publish_result = None

        false_claims = [r for r in fact_check_results if r.is_false_or_misleading]

        if self.debunk_generator.should_publish(false_claims):
            thread = self.debunk_generator.generate_thread(
                podcast_name=episode.podcast_name,
                episode_title=episode.episode_title,
                false_claims=false_claims,
                podcast_artwork_url=episode.artwork_url
            )
            logger.info(f"Generated debunk thread with {len(thread)} tweets")

            # Publish to PodDebunker account
            publish_result = self.publisher.publish_by_type(
                account_type=AccountType.DEBUNKER,
                thread=thread,
                podcast_name=episode.podcast_name,
                episode_title=episode.episode_title,
                dry_run=dry_run or self.config.dry_run
            )
            logger.info(f"Debunk publish result: {publish_result.get('status')}")
        else:
            logger.info(
                f"Skipping debunk thread - insufficient false claims "
                f"({len(false_claims)}/{self.config.factcheck.min_false_claims} required)"
            )
            publish_result = {
                "status": "skipped",
                "reason": f"insufficient_false_claims ({len(false_claims)}/{self.config.factcheck.min_false_claims})"
            }

        return fact_check_results, thread, publish_result


def run_pipeline(
    config: Config,
    episode_data: EpisodeData,
    dry_run: bool = False
) -> PipelineResult:
    """
    Convenience function to run the full pipeline on an episode.

    Args:
        config: Application configuration
        episode_data: Episode data with transcription
        dry_run: If True, don't actually publish

    Returns:
        PipelineResult with outcomes from both pipelines
    """
    orchestrator = PipelineOrchestrator(config)
    return orchestrator.process_episode(episode_data, dry_run=dry_run)
