"""AI analysis module for extracting insights from transcriptions."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
import json
from .unicode_utils import normalize_json_response

logger = logging.getLogger(__name__)


@dataclass
class PodcastHighlight:
    """Represents a key highlight from a podcast episode."""
    title: str
    insight: str
    actionable_advice: str
    relevance_score: float
    tweet_potential: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "insight": self.insight,
            "actionable_advice": self.actionable_advice,
            "relevance_score": self.relevance_score,
            "tweet_potential": self.tweet_potential
        }


class AIAnalyzer:
    """Analyzes transcriptions to extract insights and highlights."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        """
        Initialize AI analyzer.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use for analysis
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def extract_highlights(
        self, 
        transcription: str, 
        podcast_name: str,
        episode_title: str,
        max_highlights: int = 3
    ) -> List[PodcastHighlight]:
        """
        Extract top highlights from podcast transcription.
        
        Args:
            transcription: Full text transcription
            podcast_name: Name of the podcast
            episode_title: Title of the episode
            max_highlights: Maximum number of highlights to extract
            
        Returns:
            List of PodcastHighlight objects
        """
        system_prompt = """You are an expert content curator specializing in extracting 
        viral-worthy insights from podcast episodes. Your goal is to identify the most 
        compelling, actionable, and shareable moments that will resonate on social media.
        
        Focus on:
        - Counterintuitive insights
        - Practical, actionable advice
        - Memorable quotes or stories
        - Breakthrough moments or revelations
        - Controversial or thought-provoking ideas
        
        Each highlight should be tweet-worthy and spark engagement."""
        
        user_prompt = f"""Analyze this podcast transcription and extract the TOP {max_highlights} 
        most insightful, actionable highlights.
        
        Podcast: {podcast_name}
        Episode: {episode_title}
        
        Transcription:
        {transcription[:8000]}  # Limit to avoid token limits
        
        For each highlight, provide:
        1. A catchy title (5-10 words)
        2. The core insight (2-3 sentences)
        3. Actionable advice based on this insight
        4. Relevance score (0-1): How universally applicable is this?
        5. Tweet potential (0-1): How likely is this to go viral on X?
        
        Return as JSON array with these exact fields:
        [{{"title": "", "insight": "", "actionable_advice": "", "relevance_score": 0.0, "tweet_potential": 0.0}}]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            highlights_data = normalize_json_response(json.loads(content))
            
            if "highlights" in highlights_data:
                highlights_data = highlights_data["highlights"]
            elif not isinstance(highlights_data, list):
                highlights_data = [highlights_data]
            
            highlights = []
            for data in highlights_data[:max_highlights]:
                highlight = PodcastHighlight(
                    title=data.get("title", ""),
                    insight=data.get("insight", ""),
                    actionable_advice=data.get("actionable_advice", ""),
                    relevance_score=float(data.get("relevance_score", 0.5)),
                    tweet_potential=float(data.get("tweet_potential", 0.5))
                )
                highlights.append(highlight)
            
            highlights.sort(key=lambda x: x.tweet_potential, reverse=True)
            logger.info(f"Extracted {len(highlights)} highlights from {episode_title}")
            
            return highlights
            
        except Exception as e:
            logger.error(f"Error extracting highlights: {e}")
            return []
    
    def suggest_gif_theme(self, highlights: List[PodcastHighlight], episode_context: str) -> Dict[str, Any]:
        """
        Suggest relevant GIF themes for the thread.
        
        Args:
            highlights: List of extracted highlights
            episode_context: Context about the episode
            
        Returns:
            Dictionary with GIF suggestions
        """
        themes = []
        for highlight in highlights:
            prompt = f"""Based on this podcast highlight: "{highlight.title}" 
            which discusses {highlight.insight}, suggest a fresh, trendy GIF theme.
            
            Avoid overused GIFs like:
            - Mind blown
            - Mic drop
            - This is fine (burning room)
            - Drake pointing
            
            Instead suggest something:
            - Current and culturally relevant
            - Unexpected but fitting
            - From recent movies/shows/memes
            - That adds humor or emotion
            
            Return: {{"theme": "", "search_terms": [], "mood": ""}}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    response_format={"type": "json_object"}
                )
                
                suggestion = normalize_json_response(json.loads(response.choices[0].message.content))
                themes.append(suggestion)
                
            except Exception as e:
                logger.error(f"Error suggesting GIF: {e}")
                themes.append({
                    "theme": "celebration",
                    "search_terms": ["victory", "success", "winning"],
                    "mood": "triumphant"
                })
        
        return {
            "primary_theme": themes[0] if themes else None,
            "alternatives": themes[1:] if len(themes) > 1 else []
        }
    
    def score_highlights(self, highlights: List[PodcastHighlight], target_audience: str) -> List[PodcastHighlight]:
        """
        Re-score highlights based on target audience.
        
        Args:
            highlights: List of highlights to score
            target_audience: Description of target audience
            
        Returns:
            Re-scored list of highlights
        """
        for highlight in highlights:
            prompt = f"""Score this highlight's relevance for the target audience:
            
            Highlight: {highlight.title}
            Insight: {highlight.insight}
            Target Audience: {target_audience}
            
            Score from 0-1 based on:
            - Relevance to audience interests
            - Likelihood of engagement
            - Shareability within this community
            
            Return: {{"score": 0.0, "reason": ""}}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                result = normalize_json_response(json.loads(response.choices[0].message.content))
                highlight.relevance_score = float(result.get("score", highlight.relevance_score))
                
            except Exception as e:
                logger.error(f"Error scoring highlight: {e}")
        
        return sorted(highlights, key=lambda x: x.relevance_score, reverse=True)