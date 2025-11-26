"""
Podcast Summary Generator - Creates comprehensive 1000-word summaries.
Stores summaries locally in SQLite, NOT uploaded to X.com.
"""

import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import openai

logger = logging.getLogger(__name__)


@dataclass
class PodcastSummary:
    """Represents a comprehensive podcast episode summary."""
    
    episode_id: str
    podcast_name: str
    episode_title: str
    episode_number: Optional[str]
    published_date: str
    
    # The main summary content
    summary: str  # ~1000 words comprehensive summary
    word_count: int
    
    # Structured breakdown
    key_topics: List[str]  # Main topics covered
    key_takeaways: List[str]  # 5-10 actionable takeaways
    notable_quotes: List[str]  # Best quotes from the episode
    speakers_mentioned: List[str]  # Speakers/guests identified
    
    # Metadata
    created_at: str
    transcription_method: Optional[str] = None
    ai_model: str = "gpt-4"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        data['key_topics'] = json.dumps(self.key_topics)
        data['key_takeaways'] = json.dumps(self.key_takeaways)
        data['notable_quotes'] = json.dumps(self.notable_quotes)
        data['speakers_mentioned'] = json.dumps(self.speakers_mentioned)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PodcastSummary':
        """Create from dictionary (database row)."""
        # Parse JSON fields
        if isinstance(data.get('key_topics'), str):
            data['key_topics'] = json.loads(data['key_topics'])
        if isinstance(data.get('key_takeaways'), str):
            data['key_takeaways'] = json.loads(data['key_takeaways'])
        if isinstance(data.get('notable_quotes'), str):
            data['notable_quotes'] = json.loads(data['notable_quotes'])
        if isinstance(data.get('speakers_mentioned'), str):
            data['speakers_mentioned'] = json.loads(data['speakers_mentioned'])
        return cls(**data)


class PodcastSummaryDatabase:
    """SQLite database for storing podcast summaries locally."""
    
    def __init__(self, db_path: str = "data/summaries.db"):
        """
        Initialize the summary database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create the summaries table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    episode_id TEXT PRIMARY KEY,
                    podcast_name TEXT NOT NULL,
                    episode_title TEXT NOT NULL,
                    episode_number TEXT,
                    published_date TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    key_topics TEXT,
                    key_takeaways TEXT,
                    notable_quotes TEXT,
                    speakers_mentioned TEXT,
                    transcription_method TEXT,
                    ai_model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_podcast_name ON summaries (podcast_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_published_date ON summaries (published_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON summaries (created_at)")
            
            logger.info(f"Initialized summary database at {self.db_path}")
    
    def save_summary(self, summary: PodcastSummary) -> bool:
        """
        Save a podcast summary to the database.
        
        Args:
            summary: PodcastSummary object to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = summary.to_dict()
                
                conn.execute("""
                    INSERT OR REPLACE INTO summaries
                    (episode_id, podcast_name, episode_title, episode_number, 
                     published_date, summary, word_count, key_topics, key_takeaways,
                     notable_quotes, speakers_mentioned, transcription_method, ai_model,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    data['episode_id'],
                    data['podcast_name'],
                    data['episode_title'],
                    data['episode_number'],
                    data['published_date'],
                    data['summary'],
                    data['word_count'],
                    data['key_topics'],
                    data['key_takeaways'],
                    data['notable_quotes'],
                    data['speakers_mentioned'],
                    data['transcription_method'],
                    data['ai_model'],
                    data['created_at']
                ))
                
            logger.info(f"Saved summary for episode: {summary.episode_id} ({summary.word_count} words)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
            return False
    
    def get_summary(self, episode_id: str) -> Optional[PodcastSummary]:
        """
        Retrieve a summary by episode ID.
        
        Args:
            episode_id: Unique episode identifier
            
        Returns:
            PodcastSummary object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM summaries WHERE episode_id = ?",
                    (episode_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return PodcastSummary.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving summary: {e}")
            return None
    
    def summary_exists(self, episode_id: str) -> bool:
        """Check if a summary exists for the given episode."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM summaries WHERE episode_id = ?",
                (episode_id,)
            )
            return cursor.fetchone() is not None
    
    def get_all_summaries(self, 
                          podcast_name: Optional[str] = None,
                          limit: int = 100,
                          offset: int = 0) -> List[PodcastSummary]:
        """
        Get all summaries, optionally filtered by podcast.
        
        Args:
            podcast_name: Filter by podcast name (optional)
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of PodcastSummary objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if podcast_name:
                    cursor = conn.execute("""
                        SELECT * FROM summaries 
                        WHERE podcast_name = ?
                        ORDER BY published_date DESC
                        LIMIT ? OFFSET ?
                    """, (podcast_name, limit, offset))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM summaries 
                        ORDER BY published_date DESC
                        LIMIT ? OFFSET ?
                    """, (limit, offset))
                
                return [PodcastSummary.from_dict(dict(row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error retrieving summaries: {e}")
            return []
    
    def get_summary_count(self, podcast_name: Optional[str] = None) -> int:
        """Get total count of summaries."""
        with sqlite3.connect(self.db_path) as conn:
            if podcast_name:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM summaries WHERE podcast_name = ?",
                    (podcast_name,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM summaries")
            return cursor.fetchone()[0]
    
    def search_summaries(self, query: str, limit: int = 20) -> List[PodcastSummary]:
        """
        Search summaries by keyword.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching PodcastSummary objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Search in summary, title, and key topics
                cursor = conn.execute("""
                    SELECT * FROM summaries 
                    WHERE summary LIKE ? 
                       OR episode_title LIKE ?
                       OR key_topics LIKE ?
                       OR key_takeaways LIKE ?
                    ORDER BY published_date DESC
                    LIMIT ?
                """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', limit))
                
                return [PodcastSummary.from_dict(dict(row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error searching summaries: {e}")
            return []
    
    def export_to_markdown(self, episode_id: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export a summary to markdown format.
        
        Args:
            episode_id: Episode ID to export
            output_path: Optional file path to save (if None, returns string)
            
        Returns:
            Markdown string or None if error
        """
        summary = self.get_summary(episode_id)
        if not summary:
            return None
        
        md_content = f"""# {summary.episode_title}

**Podcast:** {summary.podcast_name}
**Episode:** {summary.episode_number or 'N/A'}
**Published:** {summary.published_date}
**Word Count:** {summary.word_count}

---

## Summary

{summary.summary}

---

## Key Topics

{chr(10).join(f'- {topic}' for topic in summary.key_topics)}

## Key Takeaways

{chr(10).join(f'{i+1}. {takeaway}' for i, takeaway in enumerate(summary.key_takeaways))}

## Notable Quotes

{chr(10).join(f'> "{quote}"' for quote in summary.notable_quotes)}

## Speakers Mentioned

{', '.join(summary.speakers_mentioned) if summary.speakers_mentioned else 'N/A'}

---

*Generated on {summary.created_at} using {summary.ai_model}*
"""
        
        if output_path:
            Path(output_path).write_text(md_content)
            logger.info(f"Exported summary to {output_path}")
        
        return md_content


class PodcastSummaryGenerator:
    """Generates comprehensive 1000-word summaries using AI."""
    
    def __init__(self, 
                 openai_api_key: str,
                 model: str = "gpt-4",
                 base_url: Optional[str] = None,
                 db_path: str = "data/summaries.db"):
        """
        Initialize the summary generator.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use for generation
            base_url: Optional base URL for API (e.g., OpenRouter)
            db_path: Path to summary database
        """
        if base_url:
            self.client = openai.OpenAI(api_key=openai_api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=openai_api_key)
        
        self.model = model
        self.db = PodcastSummaryDatabase(db_path)
        
        logger.info(f"Initialized PodcastSummaryGenerator with model: {model}")
    
    def generate_summary(self,
                        transcription_text: str,
                        podcast_name: str,
                        episode_title: str,
                        episode_id: str,
                        episode_number: Optional[str] = None,
                        published_date: Optional[str] = None,
                        transcription_method: Optional[str] = None) -> Optional[PodcastSummary]:
        """
        Generate a comprehensive ~1000 word summary of a podcast episode.
        
        Args:
            transcription_text: Full transcription text
            podcast_name: Name of the podcast
            episode_title: Title of the episode
            episode_id: Unique episode identifier
            episode_number: Optional episode number
            published_date: Episode publish date
            transcription_method: Method used for transcription
            
        Returns:
            PodcastSummary object or None if generation fails
        """
        # Check if summary already exists
        if self.db.summary_exists(episode_id):
            logger.info(f"Summary already exists for episode {episode_id}")
            return self.db.get_summary(episode_id)
        
        logger.info(f"Generating comprehensive summary for: {episode_title}")
        
        # Use a large chunk of the transcript for comprehensive coverage
        # GPT-4 Turbo has 128k context, so we can use a lot
        transcript_chunk = transcription_text[:120000]
        
        prompt = f"""You are an expert podcast summarizer. Create a COMPREHENSIVE, DETAILED summary of approximately 1000 words for this podcast episode.

PODCAST: {podcast_name}
EPISODE: {episode_title}
{f'EPISODE NUMBER: {episode_number}' if episode_number else ''}

TRANSCRIPT:
{transcript_chunk}

---

Create a thorough summary that:

1. **COMPREHENSIVE OVERVIEW** (~1000 words):
   - Cover ALL major topics discussed in the episode chronologically
   - Include specific details, examples, and anecdotes mentioned
   - Capture the essence of conversations and key exchanges
   - Explain complex concepts that were discussed
   - Note any debates, disagreements, or multiple perspectives
   - Include relevant context the speakers provided
   - Make it readable and engaging, not just bullet points

2. **KEY TOPICS** (5-8 main topics):
   - List the primary subjects covered in the episode

3. **KEY TAKEAWAYS** (5-10 actionable insights):
   - Practical lessons or insights listeners can apply
   - Should be specific and actionable

4. **NOTABLE QUOTES** (3-5 best quotes):
   - Memorable or impactful statements from speakers
   - Keep them concise and attribution if speaker is clear

5. **SPEAKERS MENTIONED**:
   - List any speakers, guests, or people mentioned by name

Return ONLY valid JSON in this exact format:

{{
    "summary": "Your comprehensive ~1000 word summary here...",
    "key_topics": ["Topic 1", "Topic 2", "Topic 3", ...],
    "key_takeaways": ["Takeaway 1", "Takeaway 2", ...],
    "notable_quotes": ["Quote 1", "Quote 2", ...],
    "speakers_mentioned": ["Speaker 1", "Guest Name", ...]
}}

CRITICAL REQUIREMENTS:
- The summary MUST be approximately 1000 words (800-1200 words acceptable)
- Cover the ENTIRE episode, not just the beginning
- Be specific with details, not generic
- Write in clear, engaging prose
- This summary is for archival/reference purposes, not social media
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert podcast summarizer who creates comprehensive, detailed summaries. You always return valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4,  # Balanced between creativity and accuracy
                max_tokens=4000  # Allow for long summary
            )
            
            result = response.choices[0].message.content
            data = json.loads(result)
            
            # Calculate word count
            summary_text = data.get('summary', '')
            word_count = len(summary_text.split())
            
            # Create PodcastSummary object
            summary = PodcastSummary(
                episode_id=episode_id,
                podcast_name=podcast_name,
                episode_title=episode_title,
                episode_number=episode_number,
                published_date=published_date or datetime.now().isoformat(),
                summary=summary_text,
                word_count=word_count,
                key_topics=data.get('key_topics', []),
                key_takeaways=data.get('key_takeaways', []),
                notable_quotes=data.get('notable_quotes', []),
                speakers_mentioned=data.get('speakers_mentioned', []),
                created_at=datetime.now().isoformat(),
                transcription_method=transcription_method,
                ai_model=self.model
            )
            
            # Save to database
            if self.db.save_summary(summary):
                logger.info(f"✅ Generated and saved {word_count}-word summary for: {episode_title}")
                return summary
            else:
                logger.error(f"Failed to save summary for: {episode_title}")
                return summary  # Still return the summary even if save failed
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in summary generation: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
    
    def get_summary(self, episode_id: str) -> Optional[PodcastSummary]:
        """Get an existing summary from the database."""
        return self.db.get_summary(episode_id)
    
    def list_summaries(self, 
                       podcast_name: Optional[str] = None,
                       limit: int = 100) -> List[PodcastSummary]:
        """List all summaries, optionally filtered by podcast."""
        return self.db.get_all_summaries(podcast_name=podcast_name, limit=limit)
    
    def export_summary(self, episode_id: str, output_path: Optional[str] = None) -> Optional[str]:
        """Export a summary to markdown format."""
        return self.db.export_to_markdown(episode_id, output_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get summary generation statistics."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Total summaries
                total = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
                
                # Average word count
                avg_words = conn.execute(
                    "SELECT AVG(word_count) FROM summaries"
                ).fetchone()[0] or 0
                
                # Summaries by podcast
                by_podcast = conn.execute("""
                    SELECT podcast_name, COUNT(*) as count 
                    FROM summaries 
                    GROUP BY podcast_name
                    ORDER BY count DESC
                """).fetchall()
                
                # Recent summaries
                recent = conn.execute("""
                    SELECT episode_title, podcast_name, created_at 
                    FROM summaries 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """).fetchall()
                
                return {
                    "total_summaries": total,
                    "average_word_count": round(avg_words, 0),
                    "by_podcast": {row['podcast_name']: row['count'] for row in by_podcast},
                    "recent_summaries": [
                        {"title": row['episode_title'], "podcast": row['podcast_name'], "created": row['created_at']}
                        for row in recent
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
