"""
Advanced Insight Extractor for viral content discovery.
The "brain" of the Podcasts TLDR viral content machine.
"""

import logging
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from datetime import datetime

# Rich library for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .viral_transcriber import ViralTranscriptionResult
from .unicode_utils import normalize_json_response

logger = logging.getLogger(__name__)
console = Console()


def retry_openai_call(max_retries: int = 3, base_delay: float = 2.0):
    """
    Retry decorator for OpenAI API calls.
    Handles rate limits, timeouts, and temporary failures.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except openai.RateLimitError as e:
                    # Rate limit - wait longer
                    delay = base_delay * (3 ** attempt)  # Longer backoff for rate limits
                    logger.warning(f"OpenAI rate limit hit, waiting {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    last_exception = e
                except openai.APITimeoutError as e:
                    # Timeout - retry with shorter delay
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"OpenAI timeout, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    last_exception = e
                except openai.APIConnectionError as e:
                    # Connection error - retry
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"OpenAI connection error, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    last_exception = e
                except openai.APIStatusError as e:
                    # Server error (5xx) - retry, client error (4xx) - don't retry
                    if e.status_code >= 500:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"OpenAI server error {e.status_code}, retrying in {delay}s...")
                        time.sleep(delay)
                        last_exception = e
                    else:
                        # Client error - don't retry
                        raise
                except Exception as e:
                    # Unknown error - don't retry
                    raise
            # All retries exhausted
            logger.error(f"All {max_retries} OpenAI retries failed")
            raise last_exception
        return wrapper
    return decorator


class InsightType(Enum):
    """Types of insights that can be extracted for viral content."""
    POWERFUL_QUOTE = "powerful_quote"
    SURPRISING_FACT = "surprising_fact"
    HOT_TAKE = "hot_take"
    KEY_CONCEPT = "key_concept"
    ACTIONABLE_ADVICE = "actionable_advice"
    FUNNY_MOMENT = "funny_moment"
    CONTROVERSIAL_OPINION = "controversial_opinion"
    PERSONAL_STORY = "personal_story"


@dataclass
class ViralInsight:
    """Represents a viral insight extracted from podcast content."""
    
    # Core content
    text: str
    insight_type: InsightType
    timestamp: Optional[float] = None
    speaker: Optional[str] = None
    
    # Viral metrics
    viral_score: float = 0.0
    engagement_potential: str = "medium"  # low, medium, high, viral
    tweet_formats: List[str] = None  # single, thread, poll, quote_card
    
    # Context
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    episode_context: Optional[Dict] = None
    
    # Metadata
    confidence: float = 0.0
    extraction_method: str = "ai_analysis"
    keywords: List[str] = None
    hashtags: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.tweet_formats is None:
            self.tweet_formats = ["single"]
        if self.keywords is None:
            self.keywords = []
        if self.hashtags is None:
            self.hashtags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['insight_type'] = self.insight_type.value
        return data


class ViralContentAnalyzer:
    """Analyzes content for viral potential using multiple AI techniques."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4", base_url: str = None):
        """Initialize viral content analyzer."""
        if base_url:
            self.client = openai.OpenAI(api_key=openai_api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        
        # Viral content patterns
        self.viral_patterns = {
            'emotional_triggers': [
                'shocking', 'unbelievable', 'incredible', 'amazing', 'insane',
                'mind-blowing', 'life-changing', 'game-changing', 'revolutionary'
            ],
            'curiosity_gaps': [
                'secret', 'hidden', 'truth', 'nobody tells you', 'they don\'t want you to know',
                'surprising', 'unexpected', 'turns out', 'actually'
            ],
            'social_proof': [
                'everyone', 'most people', 'successful people', 'experts', 'studies show',
                'research proves', 'millionaires', 'billionaires'
            ],
            'urgency': [
                'now', 'today', 'immediately', 'right away', 'before it\'s too late',
                'limited time', '2024', '2025', 'this year'
            ]
        }

    @retry_openai_call(max_retries=3, base_delay=2.0)
    def _call_openai_with_retry(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Make an OpenAI API call with automatic retry on failures."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def extract_viral_insights(self, 
                             transcription: ViralTranscriptionResult,
                             podcast_name: str,
                             episode_title: str,
                             max_insights: int = 10) -> List[ViralInsight]:
        """
        Extract viral insights from transcription using advanced AI analysis.
        
        Args:
            transcription: Enhanced transcription with segments and viral moments
            podcast_name: Name of the podcast
            episode_title: Episode title
            max_insights: Maximum number of insights to extract
            
        Returns:
            List of ranked viral insights
        """
        logger.info(f"Extracting viral insights from {podcast_name}: {episode_title}")
        
        insights = []
        
        # 1. Analyze pre-detected viral moments
        if transcription.viral_moments:
            insights.extend(self._analyze_viral_moments(transcription, podcast_name))
        
        # 2. Analyze full transcript for missed opportunities
        full_text_insights = self._analyze_full_transcript(
            transcription.text, podcast_name, episode_title
        )
        insights.extend(full_text_insights)
        
        # 3. Analyze segments for conversation context
        if transcription.segments:
            segment_insights = self._analyze_segments(transcription, podcast_name)
            insights.extend(segment_insights)
        
        # 4. Remove duplicates and rank by viral potential
        unique_insights = self._deduplicate_insights(insights)
        ranked_insights = self._rank_insights(unique_insights)
        
        # 5. Enhance top insights with additional metadata
        final_insights = self._enhance_insights(ranked_insights[:max_insights], transcription)

        logger.info(f"Extracted {len(final_insights)} viral insights")
        return final_insights

    def extract_key_points(self,
                          transcription: ViralTranscriptionResult,
                          podcast_name: str,
                          episode_title: str,
                          num_points: int = 5) -> List[ViralInsight]:
        """
        Extract key educational points from episode for TL;DR threads.

        This method focuses on substance and education rather than virality.
        Extracts the most important, useful, factual learnings from the entire episode.

        Args:
            transcription: Enhanced transcription with segments
            podcast_name: Name of the podcast
            episode_title: Episode title
            num_points: Number of key points to extract (default 5)

        Returns:
            List of key educational insights covering the full episode
        """
        console.print(f"[dim]  Analyzing {len(transcription.text)} chars of transcript...[/dim]")

        # Use larger chunk of transcript for comprehensive coverage
        # GPT-4 Turbo has 128k context, so we can use much more
        full_text = transcription.text[:100000]  # ~25k tokens, well within limits

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"[yellow]Extracting {num_points} key points with AI...", total=None)

            prompt = f"""
You are an educational content curator analyzing a podcast episode for its key learnings.

PODCAST: {podcast_name}
EPISODE: {episode_title}

TRANSCRIPT:
{full_text}

Extract the {num_points} most important, educational, and useful points from this ENTIRE episode.

Requirements:
1. COVER THE FULL EPISODE - Don't just extract from the beginning, ensure points span the whole conversation
2. FACTUAL AND EDUCATIONAL - Focus on what people need to know, not what's clickbait
3. STANDALONE CLARITY - Each point should make sense on its own
4. POLISHED TEXT - Remove filler words (um, uh, you know, like), make complete sentences
5. TWEET-READY - Max 280 characters per point, punchy and clear
6. ACTIONABLE - When possible, include what the listener can do with this information

For each point, provide:
- point_number: 1-{num_points}
- text: The polished, tweet-ready insight (max 280 chars)
- speaker: Who said this (if identifiable from context)
- importance: Why this matters to the audience (1 sentence)
- timestamp_estimate: Approximate timestamp in seconds (estimate from position in transcript)
- actionable: Optional action the listener can take based on this

Return ONLY valid JSON object with a "points" array of {num_points} points, sorted by their order in the episode:

{{
  "points": [
    {{
      "point_number": 1,
      "text": "Polished insight here...",
      "speaker": "Speaker Name or Unknown",
      "importance": "Why this matters...",
      "timestamp_estimate": 300,
      "actionable": "What to do with this info..."
    }}
  ]
}}

CRITICAL:
- Make text PERFECT for Twitter - no filler words, complete thoughts
- Focus on SUBSTANCE not style
- Each point should teach something valuable
- DO NOT use clickbait language or hooks
"""

            try:
                # Use retry-wrapped API call
                result = self._call_openai_with_retry(
                    system_prompt="You are an expert educational content curator who extracts key learnings from podcasts.",
                    user_prompt=prompt,
                    temperature=0.3
                )

                # Parse JSON response and normalize unicode
                try:
                    data = normalize_json_response(json.loads(result))
                except json.JSONDecodeError as json_err:
                    progress.stop()
                    console.print(f"[red]  ✗ JSON parsing error: {str(json_err)[:60]}...[/red]")
                    logger.error(f"JSON parsing error: {json_err}")
                    logger.error(f"Raw response: {result[:500]}")
                    return []

                # Handle both array and object with array responses
                if isinstance(data, dict):
                    points_data = data.get('points', data.get('key_points', []))
                else:
                    points_data = data

                if not points_data:
                    progress.stop()
                    console.print("[red]  ✗ No key points extracted from AI response[/red]")
                    logger.error(f"No key points extracted from AI response. Data structure: {type(data)}")
                    logger.error(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    return []

                # Convert to ViralInsight objects
                insights = []
                for point_data in points_data[:num_points]:
                    insight = ViralInsight(
                        text=point_data.get('text', '').strip(),
                        insight_type=InsightType.KEY_CONCEPT,  # All are educational concepts
                        timestamp=point_data.get('timestamp_estimate', 0),
                        speaker=point_data.get('speaker', 'Unknown'),
                        viral_score=0.7,  # Educational value score
                        confidence=0.9,
                        hashtags=[],
                        engagement_potential='high',  # Fixed: Changed from dict to string
                        tweet_formats=['thread'],
                        extraction_method="ai_key_points"
                    )
                    insights.append(insight)

                console.print(f"[dim]  ✓ Extracted {len(insights)} key educational points[/dim]")
                return insights

            except Exception as e:
                progress.stop()
                console.print(f"[red]  ✗ AI extraction error: {str(e)[:60]}...[/red]")
                logger.error(f"Error in AI key point extraction: {e}")
                return []

    def _analyze_viral_moments(self, 
                             transcription: ViralTranscriptionResult,
                             podcast_name: str) -> List[ViralInsight]:
        """Analyze pre-detected viral moments for insights."""
        insights = []
        
        for moment in transcription.viral_moments:
            # Get context around the viral moment
            context_text = transcription.get_text_between_timestamps(
                max(0, moment['timestamp'] - 15),
                moment['timestamp'] + 15
            )
            
            # Use AI to extract specific insight types
            insight_types = self._classify_viral_moment(moment['text'], context_text)
            
            for insight_type in insight_types:
                insight = ViralInsight(
                    text=moment['text'],
                    insight_type=insight_type,
                    timestamp=moment['timestamp'],
                    speaker=moment.get('speaker'),
                    viral_score=moment['viral_score'],
                    context_before=context_text,
                    confidence=0.8,
                    extraction_method="pre_detected"
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_full_transcript(self, 
                               text: str,
                               podcast_name: str,
                               episode_title: str) -> List[ViralInsight]:
        """Analyze full transcript using GPT-4 for comprehensive insight extraction."""
        
        prompt = f"""
You are an expert at finding viral content for social media. Analyze this podcast transcript and extract the most engaging, shareable moments.

PODCAST: {podcast_name}
EPISODE: {episode_title}

TRANSCRIPT:
{text[:8000]}  # Limit for token constraints

Extract insights in these categories:
1. POWERFUL_QUOTE - Standalone impactful statements (max 280 chars)
2. SURPRISING_FACT - Data, stats, or information that makes people stop scrolling
3. HOT_TAKE - Controversial or contrarian opinions that spark debate
4. KEY_CONCEPT - Complex topics explained simply (great for threads)
5. ACTIONABLE_ADVICE - Practical takeaways people can apply immediately
6. FUNNY_MOMENT - Humorous content that gets shared
7. PERSONAL_STORY - Relatable experiences that create connection

For each insight, provide:
- The exact text (keep it tweet-length when possible)
- Why it would go viral
- Suggested hashtags
- Best tweet format (single, thread, poll)
- Viral score (1-10)

Return as JSON array with max 8 insights, ranked by viral potential.

IMPORTANT: Focus on content that:
- Creates emotional reactions (surprise, curiosity, controversy)
- Provides clear value or entertainment
- Is highly shareable and quotable
- Appeals to broad audiences while being specific
- Uses psychological triggers for engagement
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response and normalize unicode
            try:
                ai_insights = normalize_json_response(json.loads(content))
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```(?:json)?\n?(.*?)\n?```', content, re.DOTALL)
                if json_match:
                    ai_insights = normalize_json_response(json.loads(json_match.group(1)))
                else:
                    logger.error("Failed to parse AI response as JSON")
                    return []
            
            # Convert AI insights to ViralInsight objects
            insights = []
            for ai_insight in ai_insights:
                try:
                    insight_type = InsightType(ai_insight.get('type', 'powerful_quote'))
                    
                    insight = ViralInsight(
                        text=ai_insight.get('text', ''),
                        insight_type=insight_type,
                        viral_score=ai_insight.get('viral_score', 5) / 10.0,
                        engagement_potential=self._score_to_potential(ai_insight.get('viral_score', 5)),
                        tweet_formats=ai_insight.get('formats', ['single']),
                        confidence=0.9,
                        extraction_method="ai_full_analysis",
                        keywords=ai_insight.get('keywords', []),
                        hashtags=ai_insight.get('hashtags', [])
                    )
                    
                    insights.append(insight)
                    
                except Exception as e:
                    logger.warning(f"Error parsing AI insight: {e}")
                    continue
            
            return insights
            
        except Exception as e:
            logger.error(f"Error in AI transcript analysis: {e}")
            return []
    
    def _analyze_segments(self, 
                        transcription: ViralTranscriptionResult,
                        podcast_name: str) -> List[ViralInsight]:
        """Analyze individual segments for context-aware insights."""
        insights = []
        
        # Look for conversation patterns that indicate viral content
        for i, segment in enumerate(transcription.segments):
            # Analyze segment for viral patterns
            viral_score = self._score_segment_virality(segment)
            
            if viral_score > 0.6:
                # Determine insight type based on content analysis
                insight_type = self._determine_insight_type(segment['text'])
                
                # Get surrounding context
                context_before = ""
                context_after = ""
                
                if i > 0:
                    context_before = transcription.segments[i-1].get('text', '')
                if i < len(transcription.segments) - 1:
                    context_after = transcription.segments[i+1].get('text', '')
                
                insight = ViralInsight(
                    text=segment['text'],
                    insight_type=insight_type,
                    timestamp=segment.get('start'),
                    speaker=segment.get('speaker'),
                    viral_score=viral_score,
                    context_before=context_before,
                    context_after=context_after,
                    confidence=segment.get('confidence', 0.8),
                    extraction_method="segment_analysis"
                )
                
                insights.append(insight)
        
        return insights
    
    def _classify_viral_moment(self, text: str, context: str) -> List[InsightType]:
        """Classify what type of insights a viral moment contains."""
        text_lower = text.lower()
        insight_types = []
        
        # Pattern matching for different insight types
        if any(word in text_lower for word in ['said', 'told', 'story', 'happened', 'experience']):
            insight_types.append(InsightType.PERSONAL_STORY)
        
        if any(word in text_lower for word in ['%', 'study', 'research', 'data', 'number']):
            insight_types.append(InsightType.SURPRISING_FACT)
        
        if any(word in text_lower for word in ['should', 'how to', 'key', 'secret', 'method']):
            insight_types.append(InsightType.ACTIONABLE_ADVICE)
        
        if any(word in text_lower for word in ['controversial', 'unpopular', 'disagree', 'wrong']):
            insight_types.append(InsightType.HOT_TAKE)
        
        if any(word in text_lower for word in ['funny', 'hilarious', 'laugh', 'joke']):
            insight_types.append(InsightType.FUNNY_MOMENT)
        
        # Default to powerful quote if no specific type detected
        if not insight_types:
            insight_types.append(InsightType.POWERFUL_QUOTE)
        
        return insight_types
    
    def _score_segment_virality(self, segment: Dict) -> float:
        """Score a segment for viral potential."""
        text = segment.get('text', '').lower()
        score = 0.0
        
        # Check for viral patterns
        for category, patterns in self.viral_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in text)
            if matches > 0:
                score += matches * 0.15
        
        # Length optimization (sweet spot for viral content)
        word_count = len(text.split())
        if 8 <= word_count <= 25:  # Optimal tweet length
            score += 0.3
        elif 25 <= word_count <= 50:  # Good for threads
            score += 0.2
        
        # Confidence boost
        confidence = segment.get('confidence', 0.0)
        if confidence > 0.8:
            score += 0.1
        
        return min(1.0, score)
    
    def _determine_insight_type(self, text: str) -> InsightType:
        """Determine the primary insight type for a piece of text."""
        text_lower = text.lower()
        
        # Check for specific patterns
        if re.search(r'\d+%|\d+x|\$\d+|studies?|research|data', text_lower):
            return InsightType.SURPRISING_FACT
        
        if any(word in text_lower for word in ['how to', 'you should', 'the key', 'tip', 'strategy']):
            return InsightType.ACTIONABLE_ADVICE
        
        if any(word in text_lower for word in ['controversial', 'unpopular', 'disagree', 'wrong', 'hate']):
            return InsightType.HOT_TAKE
        
        if any(word in text_lower for word in ['story', 'happened', 'experience', 'remember', 'once']):
            return InsightType.PERSONAL_STORY
        
        if any(word in text_lower for word in ['funny', 'hilarious', 'laugh', 'joke', 'crazy']):
            return InsightType.FUNNY_MOMENT
        
        if any(word in text_lower for word in ['concept', 'idea', 'understand', 'means', 'basically']):
            return InsightType.KEY_CONCEPT
        
        return InsightType.POWERFUL_QUOTE
    
    def _deduplicate_insights(self, insights: List[ViralInsight]) -> List[ViralInsight]:
        """Remove duplicate or very similar insights."""
        unique_insights = []
        seen_texts = set()
        
        for insight in insights:
            # Simple deduplication based on text similarity
            normalized_text = re.sub(r'[^\w\s]', '', insight.text.lower())
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_insights.append(insight)
        
        return unique_insights
    
    def _rank_insights(self, insights: List[ViralInsight]) -> List[ViralInsight]:
        """Rank insights by viral potential."""
        def rank_key(insight: ViralInsight) -> float:
            score = insight.viral_score
            
            # Boost certain types
            type_boosts = {
                InsightType.HOT_TAKE: 0.2,
                InsightType.SURPRISING_FACT: 0.15,
                InsightType.ACTIONABLE_ADVICE: 0.1,
                InsightType.FUNNY_MOMENT: 0.1
            }
            
            score += type_boosts.get(insight.insight_type, 0)
            
            # Boost high confidence
            score += insight.confidence * 0.1
            
            # Boost recent timestamps (more likely to be engaging)
            if insight.timestamp and insight.timestamp < 300:  # First 5 minutes
                score += 0.05
            
            return score
        
        return sorted(insights, key=rank_key, reverse=True)
    
    def _enhance_insights(self, 
                        insights: List[ViralInsight],
                        transcription: ViralTranscriptionResult) -> List[ViralInsight]:
        """Enhance insights with additional metadata and optimization."""
        
        for insight in insights:
            # Add episode context
            insight.episode_context = {
                'duration': transcription.duration,
                'method': transcription.method,
                'speakers_count': len(transcription.speakers)
            }
            
            # Optimize tweet formats based on content
            insight.tweet_formats = self._suggest_tweet_formats(insight)
            
            # Generate hashtags if not already present
            if not insight.hashtags:
                insight.hashtags = self._generate_hashtags(insight.text, insight.insight_type)
            
            # Set engagement potential
            insight.engagement_potential = self._calculate_engagement_potential(insight)
        
        return insights
    
    def _suggest_tweet_formats(self, insight: ViralInsight) -> List[str]:
        """Suggest optimal tweet formats for an insight."""
        formats = []
        
        text_length = len(insight.text)
        
        # Single tweet for short, punchy content
        if text_length <= 250:
            formats.append("single")
        
        # Thread for longer explanations
        if text_length > 150 or insight.insight_type in [InsightType.KEY_CONCEPT, InsightType.ACTIONABLE_ADVICE]:
            formats.append("thread")
        
        # Poll for controversial takes
        if insight.insight_type == InsightType.HOT_TAKE:
            formats.append("poll")
        
        # Quote card for powerful quotes
        if insight.insight_type == InsightType.POWERFUL_QUOTE and text_length <= 200:
            formats.append("quote_card")
        
        return formats if formats else ["single"]
    
    def _generate_hashtags(self, text: str, insight_type: InsightType) -> List[str]:
        """Generate relevant hashtags for the insight."""
        hashtags = ["#podcast", "#tldr"]
        
        # Type-specific hashtags
        type_hashtags = {
            InsightType.ACTIONABLE_ADVICE: ["#tips", "#productivity"],
            InsightType.SURPRISING_FACT: ["#facts", "#mindblown"],
            InsightType.HOT_TAKE: ["#hottake", "#unpopularopinion"],
            InsightType.FUNNY_MOMENT: ["#funny", "#podcastmoments"],
            InsightType.PERSONAL_STORY: ["#story", "#inspiration"],
            InsightType.KEY_CONCEPT: ["#explained", "#education"]
        }
        
        hashtags.extend(type_hashtags.get(insight_type, []))
        
        # Content-based hashtags
        text_lower = text.lower()
        content_hashtags = {
            'business': ['#business', '#entrepreneur'],
            'money': ['#money', '#finance'],
            'success': ['#success', '#motivation'],
            'technology': ['#tech', '#innovation'],
            'health': ['#health', '#wellness'],
            'psychology': ['#psychology', '#mindset']
        }
        
        for topic, tags in content_hashtags.items():
            if topic in text_lower:
                hashtags.extend(tags)
                break
        
        return hashtags[:5]  # Limit to 5 hashtags
    
    def _calculate_engagement_potential(self, insight: ViralInsight) -> str:
        """Calculate engagement potential category."""
        score = insight.viral_score
        
        if score >= 0.8:
            return "viral"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _score_to_potential(self, score: int) -> str:
        """Convert numeric score to engagement potential."""
        if score >= 8:
            return "viral"
        elif score >= 6:
            return "high"
        elif score >= 4:
            return "medium"
        else:
            return "low"


class InsightDatabase:
    """Database for storing and analyzing insights."""
    
    def __init__(self, db_path: str = "data/insights.db"):
        """Initialize insights database."""
        import sqlite3
        from pathlib import Path
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Create insights table."""
        import sqlite3
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    timestamp REAL,
                    speaker TEXT,
                    viral_score REAL NOT NULL,
                    engagement_potential TEXT,
                    tweet_formats TEXT,
                    confidence REAL,
                    hashtags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tweet_count INTEGER DEFAULT 0,
                    total_engagement INTEGER DEFAULT 0
                )
            """)
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_id ON insights (episode_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viral_score ON insights (viral_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_insight_type ON insights (insight_type)")
    
    def save_insights(self, insights: List[ViralInsight], episode_id: str) -> bool:
        """Save insights to database."""
        import sqlite3
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for insight in insights:
                    conn.execute("""
                        INSERT INTO insights 
                        (episode_id, text, insight_type, timestamp, speaker, viral_score,
                         engagement_potential, tweet_formats, confidence, hashtags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        episode_id,
                        insight.text,
                        insight.insight_type.value,
                        insight.timestamp,
                        insight.speaker,
                        insight.viral_score,
                        insight.engagement_potential,
                        ','.join(insight.tweet_formats),
                        insight.confidence,
                        ','.join(insight.hashtags)
                    ))
            
            logger.info(f"Saved {len(insights)} insights for episode {episode_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving insights: {e}")
            return False
    
    def get_top_insights(self, limit: int = 10, min_score: float = 0.5) -> List[Dict]:
        """Get top-performing insights."""
        import sqlite3
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM insights 
                WHERE viral_score >= ?
                ORDER BY viral_score DESC, total_engagement DESC
                LIMIT ?
            """, (min_score, limit))
            
            return [dict(row) for row in cursor.fetchall()]