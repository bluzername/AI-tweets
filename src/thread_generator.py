"""Thread generation module for creating X.com threads."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
import json
from .ai_analyzer import PodcastHighlight

logger = logging.getLogger(__name__)


@dataclass
class Tweet:
    """Represents a single tweet in a thread."""
    content: str
    position: int
    has_media: bool = False
    media_url: Optional[str] = None
    
    def __len__(self):
        return len(self.content)
    
    def is_valid(self):
        return 0 < len(self.content) <= 280


@dataclass
class ThreadStyle:
    """Defines the style and tone for a thread."""
    tone: str  # witty, formal, casual, inspirational, educational
    emoji_usage: str  # none, minimal, moderate, heavy
    hashtag_style: str  # none, minimal, trending, custom
    cta_style: str  # soft, medium, strong
    
    @classmethod
    def default(cls):
        return cls(
            tone="casual",
            emoji_usage="minimal",
            hashtag_style="minimal",
            cta_style="medium"
        )


class ThreadGenerator:
    """Generates X.com threads from podcast highlights."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        """
        Initialize thread generator.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_thread(
        self,
        highlights: List[PodcastHighlight],
        podcast_name: str,
        episode_title: str,
        episode_number: Optional[str],
        style: ThreadStyle,
        target_audience: str,
        podcast_handles: Dict[str, str]
    ) -> List[Tweet]:
        """
        Generate a complete thread from highlights.
        
        Args:
            highlights: List of podcast highlights
            podcast_name: Name of the podcast
            episode_title: Title of the episode
            episode_number: Episode number if available
            style: Thread style configuration
            target_audience: Description of target audience
            podcast_handles: Dictionary of X handles to tag
            
        Returns:
            List of Tweet objects forming the thread
        """
        system_prompt = f"""You are a viral content creator for X.com (Twitter). 
        Create engaging threads that get high engagement.
        
        Style Guidelines:
        - Tone: {style.tone}
        - Emoji usage: {style.emoji_usage}
        - Hashtag style: {style.hashtag_style}
        - Call-to-action strength: {style.cta_style}
        
        Target Audience: {target_audience}
        
        Thread Requirements:
        - First tweet: Hook with podcast name, episode, and irresistible teaser
        - Middle tweets: Expand on highlights with unique insights
        - Final tweet: Tag handles and strong CTA
        - Each tweet must be under 280 characters
        - Use line breaks for readability
        - Create narrative flow between tweets"""
        
        highlights_text = "\n\n".join([
            f"Highlight {i+1}: {h.title}\n"
            f"Insight: {h.insight}\n"
            f"Actionable: {h.actionable_advice}"
            for i, h in enumerate(highlights[:3])
        ])
        
        handles_text = " ".join([f"@{handle}" for handle in podcast_handles.values() if handle])
        episode_ref = f"Episode {episode_number}: {episode_title}" if episode_number else episode_title
        
        user_prompt = f"""Create a viral X thread for this podcast episode:
        
        Podcast: {podcast_name}
        Episode: {episode_ref}
        
        Key Highlights:
        {highlights_text}
        
        Handles to tag: {handles_text}
        
        Generate a thread with 5-7 tweets that:
        1. Opens with an irresistible hook
        2. Delivers value in each tweet
        3. Builds curiosity and engagement
        4. Ends with tags and strong CTA
        
        Return as JSON array of tweets:
        [{{"position": 1, "content": "tweet text"}}]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            thread_data = json.loads(content)
            
            if "tweets" in thread_data:
                thread_data = thread_data["tweets"]
            elif not isinstance(thread_data, list):
                thread_data = [thread_data]
            
            tweets = []
            for tweet_data in thread_data:
                tweet = Tweet(
                    content=tweet_data.get("content", ""),
                    position=tweet_data.get("position", len(tweets) + 1)
                )
                
                if tweet.is_valid():
                    tweets.append(tweet)
                else:
                    logger.warning(f"Invalid tweet length: {len(tweet)}")
            
            logger.info(f"Generated thread with {len(tweets)} tweets")
            return sorted(tweets, key=lambda x: x.position)
            
        except Exception as e:
            logger.error(f"Error generating thread: {e}")
            return self._generate_fallback_thread(
                podcast_name, episode_title, highlights, handles_text
            )
    
    def generate_alternative_openers(
        self,
        podcast_name: str,
        episode_title: str,
        highlights: List[PodcastHighlight],
        num_alternatives: int = 3
    ) -> List[str]:
        """
        Generate alternative opening tweets for A/B testing.
        
        Args:
            podcast_name: Name of the podcast
            episode_title: Episode title
            highlights: List of highlights
            num_alternatives: Number of alternatives to generate
            
        Returns:
            List of alternative opening tweets
        """
        prompt = f"""Generate {num_alternatives} different opening tweets for this podcast thread.
        Each should be unique in approach but equally compelling.
        
        Podcast: {podcast_name}
        Episode: {episode_title}
        Main insight: {highlights[0].title if highlights else 'Amazing insights'}
        
        Vary the hooks:
        1. Question-based hook
        2. Controversial statement
        3. Surprising statistic/fact
        4. Story/narrative opener
        5. Challenge to conventional wisdom
        
        Each must mention the podcast name and create irresistible curiosity.
        Keep under 280 characters.
        
        Return as JSON: {{"alternatives": ["tweet1", "tweet2", "tweet3"]}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            alternatives = result.get("alternatives", [])
            
            return [alt for alt in alternatives if len(alt) <= 280]
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            return [f"Just listened to {podcast_name} - {episode_title}\n\nMind = Blown 🤯\n\nThread 👇"]
    
    def adapt_thread_to_account(
        self,
        thread: List[Tweet],
        account_style: Dict[str, Any]
    ) -> List[Tweet]:
        """
        Adapt thread to specific account's style.
        
        Args:
            thread: Original thread
            account_style: Account-specific style preferences
            
        Returns:
            Adapted thread
        """
        style_prompt = f"""Rewrite this thread to match the account style:
        Tone: {account_style.get('tone', 'casual')}
        Vocabulary: {account_style.get('vocabulary', 'standard')}
        Emoji preference: {account_style.get('emoji_preference', 'minimal')}
        Signature phrases: {account_style.get('signature_phrases', [])}
        
        Maintain the core message but adapt the voice."""
        
        adapted_tweets = []
        for tweet in thread:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": style_prompt},
                        {"role": "user", "content": f"Rewrite this tweet: {tweet.content}"}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                
                adapted_content = response.choices[0].message.content.strip()
                if len(adapted_content) <= 280:
                    adapted_tweets.append(Tweet(
                        content=adapted_content,
                        position=tweet.position,
                        has_media=tweet.has_media,
                        media_url=tweet.media_url
                    ))
                else:
                    adapted_tweets.append(tweet)
                    
            except Exception as e:
                logger.error(f"Error adapting tweet: {e}")
                adapted_tweets.append(tweet)
        
        return adapted_tweets
    
    def _generate_fallback_thread(
        self,
        podcast_name: str,
        episode_title: str,
        highlights: List[PodcastHighlight],
        handles: str
    ) -> List[Tweet]:
        """Generate a basic fallback thread if AI generation fails."""
        tweets = []
        
        tweets.append(Tweet(
            content=f"🎙️ Just listened to {podcast_name}\n\n{episode_title}\n\nKey takeaways in this thread 👇",
            position=1
        ))
        
        for i, highlight in enumerate(highlights[:3], 2):
            content = f"{i-1}/ {highlight.title}\n\n{highlight.insight[:180]}..."
            tweets.append(Tweet(content=content, position=i))
        
        tweets.append(Tweet(
            content=f"That's a wrap! 🎯\n\nCheck out the full episode {handles}\n\nWhat was your biggest takeaway?",
            position=len(tweets) + 1
        ))
        
        return tweets