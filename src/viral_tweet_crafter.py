"""
Viral Tweet Crafter - The "Wordsmith" of the Podcasts TLDR machine.
Transforms insights into perfectly crafted viral tweets across multiple formats.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from datetime import datetime

# Rich library for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .viral_insight_extractor import ViralInsight, InsightType
from .unicode_utils import normalize_json_response

logger = logging.getLogger(__name__)
console = Console()


# Known podcast abbreviations for long names (>20 chars)
PODCAST_ABBREVIATIONS = {
    "The Diary Of A CEO with Steven Bartlett": "DOAC",
    "The Twenty Minute VC (20VC): Venture Capital | Startup Funding | The Pitch": "20VC",
    "Lex Fridman Podcast": "Lex Fridman",
    "The Tim Ferriss Show": "Tim Ferriss",
    "The Joe Rogan Experience": "JRE",
    "The All-In Podcast": "All-In",
}


def get_podcast_display_name(podcast_name: str, max_length: int = 20) -> str:
    """
    Get a display-friendly podcast name for tweets.

    Uses known abbreviations for long names, otherwise truncates if needed.

    Args:
        podcast_name: Full podcast name
        max_length: Maximum characters (default 20)

    Returns:
        Abbreviated or shortened podcast name
    """
    # Check known abbreviations first
    if podcast_name in PODCAST_ABBREVIATIONS:
        return PODCAST_ABBREVIATIONS[podcast_name]

    # If already short enough, return as-is
    if len(podcast_name) <= max_length:
        return podcast_name

    # Try to extract a shorter form
    # Remove common prefixes like "The"
    shortened = podcast_name
    if shortened.startswith("The "):
        shortened = shortened[4:]

    # Remove anything after "with", ":", or "|"
    for separator in [" with ", ":", "|", " - "]:
        if separator in shortened:
            shortened = shortened.split(separator)[0].strip()
            break

    # If still too long, just truncate
    if len(shortened) > max_length:
        shortened = shortened[:max_length-1].strip()

    return shortened


class TweetFormat(Enum):
    """Supported tweet formats for viral content."""
    SINGLE = "single"
    THREAD = "thread"
    POLL = "poll"
    QUOTE_CARD = "quote_card"


@dataclass
class ViralTweet:
    """Represents a crafted viral tweet."""
    
    # Core content
    content: str
    tweet_format: TweetFormat
    character_count: int = 0
    
    # Thread-specific
    thread_tweets: Optional[List[str]] = None
    thread_count: int = 1
    
    # Poll-specific
    poll_question: Optional[str] = None
    poll_options: Optional[List[str]] = None
    
    # Visual content
    quote_card_text: Optional[str] = None
    quote_card_author: Optional[str] = None
    
    # Metadata
    hashtags: List[str] = None
    mentions: List[str] = None
    timestamp_link: Optional[str] = None
    
    # Viral optimization
    hooks: List[str] = None  # Multiple hook options for A/B testing
    cta: Optional[str] = None  # Call to action
    engagement_triggers: List[str] = None
    
    # Analytics
    predicted_engagement: str = "medium"
    viral_score: float = 0.0
    target_audience: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values and calculate character count."""
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []
        if self.hooks is None:
            self.hooks = []
        if self.engagement_triggers is None:
            self.engagement_triggers = []
        
        # Calculate character count
        if self.thread_tweets:
            self.character_count = sum(len(tweet) for tweet in self.thread_tweets)
            self.thread_count = len(self.thread_tweets)
        else:
            self.character_count = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['tweet_format'] = self.tweet_format.value
        return data


class ViralHookGenerator:
    """Generates viral hooks and opening lines for tweets."""
    
    def __init__(self):
        """Initialize hook generator with proven viral patterns."""
        self.hook_patterns = {
            'curiosity': [
                "The {} that nobody talks about:",
                "What {} don't want you to know:",
                "The surprising truth about {}:",
                "Here's what happens when {}:",
                "The {} secret that changed everything:"
            ],
            'controversy': [
                "Unpopular opinion: {}",
                "Hot take: {}",
                "Controversial but true: {}",
                "Everyone's wrong about {}:",
                "The {} truth nobody wants to hear:"
            ],
            'urgency': [
                "Stop {} immediately.",
                "If you're still {}, you're behind.",
                "2025 reality check: {}",
                "Time to face facts about {}:",
                "Wake up call: {}"
            ],
            'social_proof': [
                "Successful people always {}",
                "Millionaires never {}",
                "Top performers understand {}:",
                "Industry leaders know {}:",
                "Winners always {}"
            ],
            'personal': [
                "I used to {} until I learned this:",
                "My biggest mistake was {}:",
                "What I wish I knew about {}:",
                "The {} lesson that changed my life:",
                "After 10 years, here's what I know about {}:"
            ],
            'story': [
                "A story about {}:",
                "This happened when {}:",
                "Picture this: {}",
                "Imagine if {}:",
                "Here's how {} changed everything:"
            ]
        }
    
    def generate_hooks(self, insight: ViralInsight, count: int = 3) -> List[str]:
        """Generate multiple hook options for A/B testing."""
        hooks = []
        text = insight.text
        
        # Choose patterns based on insight type
        pattern_categories = self._select_hook_categories(insight.insight_type)
        
        for category in pattern_categories:
            patterns = self.hook_patterns.get(category, [])
            for pattern in patterns[:count]:
                try:
                    # Extract key concept for hook
                    key_concept = self._extract_key_concept(text)
                    hook = pattern.format(key_concept)
                    hooks.append(hook)
                except:
                    continue
        
        # Add direct quotes as hooks
        if len(text) <= 100:
            hooks.append(f'"{text}"')
        
        return hooks[:count]
    
    def _select_hook_categories(self, insight_type: InsightType) -> List[str]:
        """Select appropriate hook categories for insight type."""
        type_mapping = {
            InsightType.HOT_TAKE: ['controversy', 'urgency'],
            InsightType.SURPRISING_FACT: ['curiosity', 'social_proof'],
            InsightType.ACTIONABLE_ADVICE: ['social_proof', 'personal'],
            InsightType.PERSONAL_STORY: ['personal', 'story'],
            InsightType.FUNNY_MOMENT: ['story', 'personal'],
            InsightType.KEY_CONCEPT: ['curiosity', 'social_proof'],
            InsightType.POWERFUL_QUOTE: ['personal', 'story']
        }
        
        return type_mapping.get(insight_type, ['curiosity', 'personal'])
    
    def _extract_key_concept(self, text: str) -> str:
        """Extract key concept from text for hook generation."""
        # Simple keyword extraction
        text_lower = text.lower()
        
        # Common concepts that work well in hooks
        concepts = [
            'productivity', 'success', 'money', 'business', 'leadership',
            'psychology', 'mindset', 'habits', 'marketing', 'sales',
            'entrepreneurship', 'investing', 'creativity', 'innovation'
        ]
        
        for concept in concepts:
            if concept in text_lower:
                return concept
        
        # Fallback: use first meaningful word
        words = text.split()
        meaningful_words = [w for w in words if len(w) > 4 and w.lower() not in ['this', 'that', 'what', 'when', 'where', 'they', 'them']]
        
        return meaningful_words[0] if meaningful_words else 'this topic'


class ThreadBuilder:
    """Builds engaging tweet threads from insights."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4", base_url: str = None):
        """Initialize thread builder."""
        if base_url:
            self.client = openai.OpenAI(api_key=openai_api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model

    def build_educational_thread(
        self,
        key_points: List,  # List of ViralInsight objects
        episode_title: str,
        podcast_name: str,
        podcast_handle: str = None,
        host_handles: List[str] = None,
        guest_handles: List[str] = None,
        episode_number: str = None
    ) -> List[str]:
        """
        Build a 6-tweet educational thread summarizing a podcast episode.

        Tweet 1: Attribution with all handles
        Tweets 2-6: The 5 key educational points

        Args:
            key_points: List of 5 key educational points (ViralInsight objects)
            episode_title: Title of the episode
            podcast_name: Name of the podcast
            podcast_handle: Twitter handle of the podcast
            host_handles: List of host Twitter handles
            guest_handles: List of guest Twitter handles/names

        Returns:
            List of 6 tweet strings forming a complete thread
        """
        if not key_points or len(key_points) < 5:
            logger.error(f"Need 5 key points for thread, got {len(key_points) if key_points else 0}")
            return []

        if host_handles is None:
            host_handles = []
        if guest_handles is None:
            guest_handles = []

        # Format points for the prompt (handle both ViralInsight objects and plain strings)
        points_text = "\n".join([
            f"{i+1}. {point.text if hasattr(point, 'text') else point}"
            for i, point in enumerate(key_points[:5])
        ])

        # Format handles
        all_handles = []
        if podcast_handle:
            all_handles.append(podcast_handle)
        all_handles.extend(host_handles)
        all_handles.extend(guest_handles)

        handles_text = " ".join(all_handles) if all_handles else podcast_name

        # Add episode number to title if available
        title_with_ep = episode_title
        if episode_number:
            title_with_ep = f"[Ep {episode_number}] {episode_title}"

        # Get abbreviated podcast name for display
        podcast_display = get_podcast_display_name(podcast_name)

        # Format episode info for prompt
        if episode_number:
            ep_instruction = f"Include episode number after the emoji: [Ep {episode_number}]"
            ep_example = f'"{podcast_display} 🎙️ [Ep {episode_number}] Great insights from @Handle on [topic]!"'
        else:
            ep_instruction = "No episode number available - do NOT include any episode brackets"
            ep_example = f'"{podcast_display} 🎙️ Great insights from @Handle on [topic]!"'

        prompt = f"""
Create a 6-tweet educational thread summarizing this podcast episode.

PODCAST: {podcast_name}
EPISODE TITLE: {episode_title}
HANDLES TO TAG: {handles_text}

KEY POINTS FROM EPISODE:
{points_text}

Generate EXACTLY 6 tweets:

TWEET 1 (Attribution):
- MUST start with the podcast name: "{podcast_display}"
- Follow with the podcast emoji 🎙️
- {ep_instruction}
- Tag the podcast and host using the HANDLES listed above
- Brief 1-sentence summary of what the episode covers
- Engaging but professional tone
- Example format: {ep_example}
- Max 280 chars
- NO links or URLs

TWEETS 2-6 (Key Points):
- One tweet per key point, in order
- Number with emoji: 1️⃣, 2️⃣, 3️⃣, 4️⃣, 5️⃣
- Factual, educational, clear
- Each tweet standalone understandable
- NO clickbait language
- NO rhetorical hooks
- NO "you won't believe" type language
- Include speaker attribution if it adds value (e.g., "According to @handle...")
- Max 270 chars each (accounting for numbering)
- Minimal emoji use (only if genuinely helpful)
- Maximum 2 relevant hashtags in final tweet only

X.COM ALGORITHM OPTIMIZATION RULES:
- Use line breaks for readability (easier to read = more engagement)
- Ask a question in final tweet if natural (questions get replies)
- Use @mentions naturally (tagged users may engage/retweet)
- Keep sentences punchy and scannable
- Use active voice (more engaging than passive)
- Include specific numbers/data when available (stops scrolling)
- NO external links (they reduce reach)
- Make first line of each tweet compelling (shows in timeline)

CRITICAL REQUIREMENTS:
- All tweets must be complete, polished sentences
- NO filler words (um, uh, you know, like)
- Focus on SUBSTANCE and EDUCATION
- Make each point valuable and memorable
- Ensure proper Twitter handle format (@username)

Return as JSON object with format:
{{
  "thread": [
    "Tweet 1 text...",
    "Tweet 2 text...",
    "Tweet 3 text...",
    "Tweet 4 text...",
    "Tweet 5 text...",
    "Tweet 6 text..."
  ]
}}
"""

        console.print(f"[dim]  Crafting 6-tweet thread with AI...[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[yellow]Generating educational thread...", total=None)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at creating clear, educational Twitter threads that summarize podcast episodes."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.4  # Lower for more consistent, factual output
                )

                content = response.choices[0].message.content

                # Parse JSON response and normalize unicode
                data = normalize_json_response(json.loads(content))
                thread_tweets = data.get('thread', [])

                if len(thread_tweets) != 6:
                    progress.stop()
                    console.print(f"[yellow]  ⚠ Expected 6 tweets, got {len(thread_tweets)}, using fallback[/yellow]")
                    logger.error(f"Expected 6 tweets, got {len(thread_tweets)}")
                    return self._create_fallback_educational_thread(
                        key_points, episode_title, podcast_name, handles_text
                    )

                # Validate each tweet
                validated_thread = []
                for i, tweet in enumerate(thread_tweets):
                    if not isinstance(tweet, str):
                        progress.stop()
                        console.print(f"[yellow]  ⚠ Invalid tweet format, using fallback[/yellow]")
                        logger.error(f"Tweet {i+1} is not a string")
                        return self._create_fallback_educational_thread(
                            key_points, episode_title, podcast_name, handles_text
                        )

                    # Check length
                    if len(tweet) > 280:
                        logger.warning(f"Tweet {i+1} exceeds 280 chars ({len(tweet)}), truncating")
                        tweet = tweet[:277] + "..."

                    validated_thread.append(tweet.strip())

                # Ensure first tweet starts with podcast name
                if validated_thread and not validated_thread[0].startswith(podcast_display):
                    first_tweet = validated_thread[0]
                    # If it starts with emoji, prepend podcast name
                    if first_tweet.startswith("🎙️"):
                        validated_thread[0] = f"{podcast_display} {first_tweet}"
                    else:
                        # Prepend podcast name + emoji
                        validated_thread[0] = f"{podcast_display} 🎙️ {first_tweet}"
                    # Re-check length after modification
                    if len(validated_thread[0]) > 280:
                        validated_thread[0] = validated_thread[0][:277] + "..."
                    logger.debug(f"Prepended podcast name to first tweet: {podcast_display}")

                console.print(f"[dim]  ✓ Created thread with {len(validated_thread)} tweets[/dim]")
                return validated_thread

            except Exception as e:
                progress.stop()
                console.print(f"[red]  ✗ Thread generation error: {str(e)[:60]}...[/red]")
                logger.error(f"Error creating educational thread: {e}")
                return self._create_fallback_educational_thread(
                    key_points, episode_title, podcast_name, handles_text
                )

    def _create_fallback_educational_thread(
        self,
        key_points: List,
        episode_title: str,
        podcast_name: str,
        handles_text: str,
        episode_number: str = None
    ) -> List[str]:
        """
        Create a simple fallback thread if AI generation fails.

        Args:
            key_points: List of key points
            episode_title: Episode title
            podcast_name: Podcast name
            handles_text: Formatted handles string
            episode_number: Optional episode number

        Returns:
            List of 6 tweet strings
        """
        thread = []

        # Get abbreviated podcast name for display
        podcast_display = get_podcast_display_name(podcast_name)

        # Tweet 1: Attribution with podcast name prefix and episode number
        ep_prefix = f"[Ep {episode_number}] " if episode_number else ""
        thread.append(
            f"{podcast_display} 🎙️ {ep_prefix}Key insights from {handles_text} on {episode_title[:80]}..."
        )

        # Tweets 2-6: Key points
        emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
        for i, point in enumerate(key_points[:5]):
            tweet = f"{emoji_numbers[i]} {point.text[:270]}"
            thread.append(tweet)

        return thread
    
    def build_thread(self, insight: ViralInsight, podcast_name: str, episode_title: str) -> List[str]:
        """Build an engaging tweet thread from an insight."""
        
        prompt = f"""
You are an expert at creating viral Twitter threads. Transform this podcast insight into an engaging, highly shareable thread.

PODCAST: {podcast_name}
EPISODE: {episode_title}
INSIGHT TYPE: {insight.insight_type.value}
INSIGHT: {insight.text}

Create a thread that:
1. HOOKS immediately with curiosity/controversy
2. DELIVERS value in digestible chunks
3. BUILDS momentum with each tweet
4. ENDS with strong engagement (question/CTA)

RULES:
- Each tweet max 270 chars (leave room for numbering)
- Use emojis strategically for engagement
- Include relevant hashtags in final tweet
- Make each tweet valuable standalone
- Use psychological triggers (scarcity, social proof, etc.)
- End with question or call to action
- DO NOT add tweet numbers or "1/" "2/" prefixes - they will be added automatically

THREAD STRUCTURE:
Tweet 1: Hook tweet (curiosity/controversy)
Tweets 2-4: Value delivery (break down the insight)
Tweet 5: Context/credibility (podcast source)
Tweet 6: Engagement closer (question/CTA + hashtags)

Return as JSON array of tweet strings WITHOUT numbering prefixes.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response and normalize unicode
            try:
                thread_tweets = normalize_json_response(json.loads(content))
            except json.JSONDecodeError:
                # Try to extract JSON from markdown
                json_match = re.search(r'```(?:json)?\n?(.*?)\n?```', content, re.DOTALL)
                if json_match:
                    thread_tweets = normalize_json_response(json.loads(json_match.group(1)))
                else:
                    # Fallback: create basic thread
                    return self._create_basic_thread(insight)
            
            # Validate and clean thread
            cleaned_thread = []
            for i, tweet in enumerate(thread_tweets[:8]):  # Max 8 tweets
                if isinstance(tweet, str):
                    # Strip any existing numbering patterns (1/, 2/, etc.)
                    tweet = re.sub(r'^\s*\d+[/)]\s*', '', tweet.strip())

                    # Check length after stripping numbering
                    if len(tweet) <= 270:
                        cleaned_thread.append(tweet)

            return cleaned_thread if cleaned_thread else self._create_basic_thread(insight)
            
        except Exception as e:
            logger.error(f"Error creating AI thread: {e}")
            return self._create_basic_thread(insight)
    
    def _create_basic_thread(self, insight: ViralInsight) -> List[str]:
        """Create a basic thread structure as fallback."""
        text = insight.text
        
        # Split into sentences for thread structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        thread = []
        
        # Hook tweet
        hook = f"1/ Interesting insight from a recent podcast:"
        thread.append(hook)
        
        # Content tweets
        for i, sentence in enumerate(sentences[:3], 2):
            if len(sentence) > 250:
                # Split long sentences
                words = sentence.split()
                mid = len(words) // 2
                part1 = ' '.join(words[:mid])
                part2 = ' '.join(words[mid:])
                thread.append(f"{i}/ {part1}...")
                thread.append(f"{i+1}/ ...{part2}")
            else:
                thread.append(f"{i}/ {sentence}")
        
        # Engagement closer
        closer_num = len(thread) + 1
        thread.append(f"{closer_num}/ What's your take on this? 🤔 #podcast #insights")
        
        return thread


class PollGenerator:
    """Generates engaging polls from controversial insights."""
    
    def create_poll(self, insight: ViralInsight) -> Tuple[str, List[str]]:
        """Create a poll from a controversial insight."""
        
        text = insight.text.lower()
        
        # Generate question based on content
        if 'better' in text or 'worse' in text:
            question = f"What's your take on this insight? 🤔"
            options = ["Completely agree 💯", "Partially agree 🤷", "Disagree entirely ❌", "Need more context 📚"]
        
        elif 'should' in text or 'shouldn\'t' in text:
            question = f"Do you agree with this advice?"
            options = ["Yes, absolutely! ✅", "Depends on context 🤔", "No, bad advice ❌", "I'm not sure 🤷"]
        
        elif any(word in text for word in ['controversial', 'unpopular', 'wrong', 'right']):
            question = f"Hot take alert! 🔥 Your thoughts?"
            options = ["Love this take 🔥", "Interesting point 🤔", "Strongly disagree ❌", "It's complicated 🤷"]
        
        else:
            # Generic poll structure
            question = f"What's your experience with this? 💭"
            options = ["Totally relatable 💯", "Somewhat true 🤔", "Not my experience ❌", "Never thought about it 💭"]
        
        return question, options


class ViralTweetCrafter:
    """Main tweet crafting engine that creates viral content across formats."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4", base_url: str = None):
        """Initialize viral tweet crafter."""
        if base_url:
            self.client = openai.OpenAI(api_key=openai_api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model

        # Initialize components
        self.hook_generator = ViralHookGenerator()
        self.thread_builder = ThreadBuilder(openai_api_key, model, base_url=base_url)
        self.poll_generator = PollGenerator()

        logger.info("ViralTweetCrafter initialized")
    
    def craft_viral_tweets(self, 
                         insight: ViralInsight,
                         podcast_name: str,
                         episode_title: str,
                         podcast_handles: Dict[str, Any] = None,
                         formats: List[str] = None) -> List[ViralTweet]:
        """
        Craft viral tweets in multiple formats from an insight.
        
        Args:
            insight: The viral insight to craft tweets from
            podcast_name: Name of the podcast
            episode_title: Episode title
            podcast_handles: Podcast and creator social handles
            formats: Specific formats to create (if None, use insight's suggested formats)
            
        Returns:
            List of crafted viral tweets in different formats
        """
        logger.info(f"Crafting viral tweets for insight: {insight.insight_type.value}")
        
        if formats is None:
            formats = insight.tweet_formats
        
        tweets = []
        handles = podcast_handles or {}
        
        for format_name in formats:
            try:
                tweet_format = TweetFormat(format_name)
                
                if tweet_format == TweetFormat.SINGLE:
                    tweet = self._craft_single_tweet(insight, podcast_name, handles)
                
                elif tweet_format == TweetFormat.THREAD:
                    tweet = self._craft_thread_tweet(insight, podcast_name, episode_title, handles)
                
                elif tweet_format == TweetFormat.POLL:
                    tweet = self._craft_poll_tweet(insight, podcast_name, handles)
                
                elif tweet_format == TweetFormat.QUOTE_CARD:
                    tweet = self._craft_quote_card_tweet(insight, podcast_name, handles)
                
                else:
                    continue
                
                if tweet:
                    tweets.append(tweet)
                    
            except Exception as e:
                logger.error(f"Error crafting {format_name} tweet: {e}")
                continue
        
        # Sort by predicted viral potential
        tweets.sort(key=lambda t: t.viral_score, reverse=True)
        
        logger.info(f"Crafted {len(tweets)} viral tweets")
        return tweets
    
    def _craft_single_tweet(self, 
                          insight: ViralInsight,
                          podcast_name: str,
                          handles: Dict) -> ViralTweet:
        """Craft a single viral tweet."""
        
        # Generate multiple hooks for A/B testing
        hooks = self.hook_generator.generate_hooks(insight, 3)
        
        # Main content
        text = insight.text
        if len(text) > 200:
            text = text[:197] + "..."
        
        # Build tweet components
        content_options = []
        
        for hook in hooks:
            tweet_content = f"{hook}\n\n{text}"
            
            # Add timestamp link if available
            timestamp_part = ""
            if insight.timestamp:
                minutes = int(insight.timestamp // 60)
                seconds = int(insight.timestamp % 60)
                timestamp_part = f"\n\n⏰ {minutes}:{seconds:02d}"
            
            # Add mentions
            mentions_part = ""
            if handles.get('podcast'):
                mentions_part = f"\n\nFrom @{handles['podcast']}"
            
            # Add hashtags
            hashtags_part = f"\n\n{' '.join(insight.hashtags[:3])}"
            
            # Combine all parts
            full_content = tweet_content + timestamp_part + mentions_part + hashtags_part
            
            # Check length
            if len(full_content) <= 280:
                content_options.append(full_content)
        
        # Select best option (shortest that fits)
        if content_options:
            final_content = content_options[0]
        else:
            # Fallback: just the insight text with minimal formatting
            final_content = f"{text}\n\n{' '.join(insight.hashtags[:2])}"
        
        return ViralTweet(
            content=final_content,
            tweet_format=TweetFormat.SINGLE,
            hooks=hooks,
            hashtags=insight.hashtags,
            mentions=[handles.get('podcast')] if handles.get('podcast') else [],
            timestamp_link=f"⏰ {int(insight.timestamp//60)}:{int(insight.timestamp%60):02d}" if insight.timestamp else None,
            predicted_engagement=insight.engagement_potential,
            viral_score=insight.viral_score,
            engagement_triggers=self._identify_engagement_triggers(insight.text)
        )
    
    def _craft_thread_tweet(self,
                          insight: ViralInsight,
                          podcast_name: str,
                          episode_title: str,
                          handles: Dict) -> ViralTweet:
        """Craft a tweet thread."""
        
        # Build thread using AI
        thread_tweets = self.thread_builder.build_thread(insight, podcast_name, episode_title)
        
        # Add podcast attribution to final tweet
        if thread_tweets and handles.get('podcast'):
            final_tweet = thread_tweets[-1]
            if len(final_tweet) < 200:  # Room for attribution
                thread_tweets[-1] = f"{final_tweet}\n\nFrom @{handles['podcast']}"
        
        # Generate hooks for the first tweet
        hooks = self.hook_generator.generate_hooks(insight, 3)
        
        return ViralTweet(
            content=thread_tweets[0] if thread_tweets else "Thread about viral insight",
            tweet_format=TweetFormat.THREAD,
            thread_tweets=thread_tweets,
            hooks=hooks,
            hashtags=insight.hashtags,
            mentions=[handles.get('podcast')] if handles.get('podcast') else [],
            timestamp_link=f"⏰ {int(insight.timestamp//60)}:{int(insight.timestamp%60):02d}" if insight.timestamp else None,
            predicted_engagement=insight.engagement_potential,
            viral_score=insight.viral_score * 1.2,  # Threads often get more engagement
            engagement_triggers=self._identify_engagement_triggers(insight.text)
        )
    
    def _craft_poll_tweet(self,
                        insight: ViralInsight,
                        podcast_name: str,
                        handles: Dict) -> ViralTweet:
        """Craft a poll tweet."""
        
        question, options = self.poll_generator.create_poll(insight)
        
        # Create poll setup text
        setup_text = f"From a recent {podcast_name} episode:\n\n\"{insight.text[:150]}{'...' if len(insight.text) > 150 else ''}\"\n\n{question}"
        
        # Add attribution
        if handles.get('podcast'):
            setup_text += f"\n\n@{handles['podcast']}"
        
        return ViralTweet(
            content=setup_text,
            tweet_format=TweetFormat.POLL,
            poll_question=question,
            poll_options=options,
            hashtags=insight.hashtags,
            mentions=[handles.get('podcast')] if handles.get('podcast') else [],
            predicted_engagement="high",  # Polls typically get high engagement
            viral_score=insight.viral_score * 1.1,
            engagement_triggers=['poll_interaction'] + self._identify_engagement_triggers(insight.text)
        )
    
    def _craft_quote_card_tweet(self,
                              insight: ViralInsight,
                              podcast_name: str,
                              handles: Dict) -> ViralTweet:
        """Craft a quote card tweet."""
        
        # Prepare quote text (limit for visual readability)
        quote_text = insight.text
        if len(quote_text) > 180:
            quote_text = quote_text[:177] + "..."
        
        # Determine quote author
        author = insight.speaker or "Podcast Guest"
        if author == "speaker_1":
            author = "Host"
        elif author == "speaker_2":
            author = "Guest"
        
        # Tweet text accompanying the quote card
        tweet_text = f"💡 Quote of the day from {podcast_name}:\n\n[Quote Card: See image]\n\n{' '.join(insight.hashtags[:3])}"
        
        # Add attribution
        if handles.get('podcast'):
            tweet_text += f"\n\n@{handles['podcast']}"
        
        return ViralTweet(
            content=tweet_text,
            tweet_format=TweetFormat.QUOTE_CARD,
            quote_card_text=quote_text,
            quote_card_author=f"— {author}",
            hashtags=insight.hashtags,
            mentions=[handles.get('podcast')] if handles.get('podcast') else [],
            predicted_engagement=insight.engagement_potential,
            viral_score=insight.viral_score * 1.05,  # Quote cards can be visually engaging
            engagement_triggers=['visual_content'] + self._identify_engagement_triggers(insight.text)
        )
    
    def _identify_engagement_triggers(self, text: str) -> List[str]:
        """Identify psychological triggers that drive engagement."""
        triggers = []
        text_lower = text.lower()
        
        # Emotional triggers
        if any(word in text_lower for word in ['shocking', 'amazing', 'incredible', 'unbelievable']):
            triggers.append('emotional_reaction')
        
        # Curiosity triggers
        if any(word in text_lower for word in ['secret', 'hidden', 'truth', 'nobody knows']):
            triggers.append('curiosity_gap')
        
        # Social proof triggers
        if any(word in text_lower for word in ['successful', 'experts', 'research', 'studies']):
            triggers.append('social_proof')
        
        # Controversy triggers
        if any(word in text_lower for word in ['controversial', 'disagree', 'wrong', 'unpopular']):
            triggers.append('controversy')
        
        # Value triggers
        if any(word in text_lower for word in ['how to', 'tips', 'advice', 'method', 'system']):
            triggers.append('practical_value')
        
        return triggers
    
    def optimize_for_engagement(self, tweets: List[ViralTweet]) -> List[ViralTweet]:
        """Optimize tweets for maximum engagement."""
        
        for tweet in tweets:
            # Add engagement optimization
            tweet.cta = self._generate_cta(tweet)
            
            # Optimize timing suggestions could be added here
            # Optimize hashtag selection
            tweet.hashtags = self._optimize_hashtags(tweet.hashtags, tweet.tweet_format)
        
        return tweets
    
    def _generate_cta(self, tweet: ViralTweet) -> str:
        """Generate call-to-action based on tweet type."""
        
        cta_options = {
            TweetFormat.SINGLE: [
                "What's your take? 🤔",
                "Agree or disagree? 💭",
                "Have you experienced this? 👇",
                "Thoughts? 💬"
            ],
            TweetFormat.THREAD: [
                "What did you think of this thread? 🧵",
                "Which point resonated most? 💭",
                "Have you tried this approach? 👇",
                "Share your experience! 💬"
            ],
            TweetFormat.POLL: [
                "Vote and share your thoughts! 🗳️",
                "Curious about your experience! 💭",
                "Let's see what everyone thinks! 👇"
            ],
            TweetFormat.QUOTE_CARD: [
                "What's your favorite quote? 💭",
                "Does this resonate with you? 🤔",
                "Share if you agree! 👇"
            ]
        }
        
        options = cta_options.get(tweet.tweet_format, ["Thoughts? 💬"])
        return options[0]  # Could randomize for A/B testing
    
    def _optimize_hashtags(self, hashtags: List[str], format_type: TweetFormat) -> List[str]:
        """Optimize hashtags for specific tweet format."""
        
        # Format-specific hashtag optimization
        format_hashtags = {
            TweetFormat.THREAD: ["#thread", "#🧵"],
            TweetFormat.POLL: ["#poll", "#vote"],
            TweetFormat.QUOTE_CARD: ["#quote", "#wisdom"]
        }
        
        optimized = hashtags[:3]  # Start with top 3 content hashtags
        
        # Add format-specific hashtags
        if format_type in format_hashtags:
            for tag in format_hashtags[format_type]:
                if tag not in optimized and len(optimized) < 5:
                    optimized.append(tag)
        
        return optimized