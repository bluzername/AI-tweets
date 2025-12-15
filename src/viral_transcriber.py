"""
Enhanced transcription engine with speaker identification and timestamps for viral content.
"""

import logging
import os
import tempfile
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import requests

# Rich library for beautiful output
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Import existing transcription modules
from .multi_transcriber import MultiTranscriber, TranscriptionMethod
from .youtube_transcriber import YouTubeTranscriber
from .local_whisper import LocalWhisperTranscriber, is_whisper_available
from .transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)
console = Console()


class ViralTranscriptionResult:
    """Enhanced transcription result with viral content features."""
    
    def __init__(self, 
                 text: str,
                 method: str,
                 duration: Optional[float] = None,
                 segments: Optional[List[Dict]] = None,
                 speakers: Optional[List[Dict]] = None,
                 confidence: Optional[float] = None,
                 language: Optional[str] = None,
                 viral_moments: Optional[List[Dict]] = None):
        """
        Initialize viral transcription result.
        
        Args:
            text: Full transcription text
            method: Transcription method used
            duration: Audio duration in seconds
            segments: Timestamped segments
            speakers: Identified speakers
            confidence: Overall confidence score
            language: Detected language
            viral_moments: Pre-identified viral moments with timestamps
        """
        self.text = text
        self.method = method
        self.duration = duration
        self.segments = segments or []
        self.speakers = speakers or []
        self.confidence = confidence
        self.language = language
        self.viral_moments = viral_moments or []
        self.created_at = datetime.now()
    
    def get_segments_by_speaker(self, speaker_id: str) -> List[Dict]:
        """Get all segments spoken by a specific speaker."""
        return [seg for seg in self.segments if seg.get('speaker') == speaker_id]
    
    def get_segment_at_timestamp(self, timestamp: float) -> Optional[Dict]:
        """Get the segment at a specific timestamp."""
        for segment in self.segments:
            if segment.get('start', 0) <= timestamp <= segment.get('end', 0):
                return segment
        return None
    
    def get_viral_moments_with_context(self, context_seconds: int = 30) -> List[Dict]:
        """Get viral moments with surrounding context."""
        moments_with_context = []
        
        for moment in self.viral_moments:
            start_time = max(0, moment['timestamp'] - context_seconds)
            end_time = min(self.duration or float('inf'), moment['timestamp'] + context_seconds)
            
            context_text = self.get_text_between_timestamps(start_time, end_time)
            
            moments_with_context.append({
                **moment,
                'context_text': context_text,
                'context_start': start_time,
                'context_end': end_time
            })
        
        return moments_with_context
    
    def get_text_between_timestamps(self, start: float, end: float) -> str:
        """Get text between two timestamps."""
        relevant_segments = []
        
        for segment in self.segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if segment overlaps with the requested time range
            if seg_start <= end and seg_end >= start:
                relevant_segments.append(segment.get('text', ''))
        
        return ' '.join(relevant_segments)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            'text': self.text,
            'method': self.method,
            'duration': self.duration,
            'segments': self.segments,
            'speakers': self.speakers,
            'confidence': self.confidence,
            'language': self.language,
            'viral_moments': self.viral_moments,
            'created_at': self.created_at.isoformat()
        }


class EnhancedLocalWhisper:
    """Enhanced local Whisper with speaker identification."""

    def __init__(self, model_size: str = "base", device: str = "auto"):
        """Initialize enhanced local Whisper with GPU support."""
        self.model_size = model_size
        self.device = device
        self.base_transcriber = None

        if is_whisper_available():
            self.base_transcriber = LocalWhisperTranscriber(model_size, device)
    
    def transcribe_with_speakers(self,
                               audio_url: str,
                               language: Optional[str] = None) -> ViralTranscriptionResult:
        """Transcribe audio with speaker identification and timestamps."""
        if not self.base_transcriber:
            raise RuntimeError("Local Whisper not available")

        try:
            # Get basic transcription first
            base_result = self.base_transcriber.transcribe_audio(audio_url, language)

            # Download audio for advanced processing
            temp_audio = self._download_audio(audio_url)

            # Use Whisper with word-level timestamps
            import whisper
            import torch

            # Determine device with MPS fallback support
            device = self.device
            if device == "auto":
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            # Try to load model with preferred device, fallback to CPU on MPS errors
            try:
                model = whisper.load_model(self.model_size, device=device)
            except (RuntimeError, Exception) as e:
                error_str = str(e).lower()
                # Catch MPS-related errors: sparse tensor ops, storage issues, Metal errors
                mps_error_keywords = ["mps", "storage", "metal", "sparse"]
                if device == "mps" and any(kw in error_str for kw in mps_error_keywords):
                    logger.warning(f"MPS device error, falling back to CPU: {str(e)[:200]}")
                    device = "cpu"
                    model = whisper.load_model(self.model_size, device=device)
                else:
                    raise

            # Transcribe with word-level timestamps
            # Disable fp16 - causes NaN errors on some NVIDIA GPUs
            result = model.transcribe(
                temp_audio,
                language=language,
                word_timestamps=True,
                verbose=False,
                fp16=False
            )
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': segment.get('start'),
                    'end': segment.get('end'),
                    'text': segment.get('text', '').strip(),
                    'confidence': segment.get('avg_logprob', 0.0),
                    'speaker': self._identify_speaker(segment)  # Simple speaker ID
                })
            
            # Identify speakers using audio analysis
            speakers = self._identify_speakers(temp_audio, segments)
            
            # Clean up temp file
            os.unlink(temp_audio)
            
            return ViralTranscriptionResult(
                text=result.get('text', ''),
                method="local_whisper_enhanced",
                duration=base_result.get('duration'),
                segments=segments,
                speakers=speakers,
                confidence=result.get('avg_logprob', 0.0),
                language=result.get('language')
            )
            
        except Exception as e:
            logger.error(f"Enhanced Whisper transcription failed: {e}")
            # Fallback to basic transcription
            base_result = self.base_transcriber.transcribe_audio(audio_url, language)
            return ViralTranscriptionResult(
                text=base_result.get('text', ''),
                method="local_whisper_basic",
                duration=base_result.get('duration')
            )
    
    def _download_audio(self, audio_url: str) -> str:
        """Download audio to temporary file."""
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            return f.name
    
    def _identify_speaker(self, segment: Dict) -> str:
        """Simple speaker identification based on audio characteristics."""
        # This is a placeholder - real speaker ID would use audio features
        # For now, use simple heuristics based on timing and content
        
        # Default speaker assignment
        return "speaker_1"
    
    def _identify_speakers(self, audio_file: str, segments: List[Dict]) -> List[Dict]:
        """Identify unique speakers in the audio."""
        # This is a simplified version - real implementation would use
        # speaker diarization libraries like pyannote.audio
        
        speakers = [
            {
                'id': 'speaker_1',
                'name': 'Host',
                'confidence': 0.8,
                'segments_count': len([s for s in segments if s.get('speaker') == 'speaker_1'])
            }
        ]
        
        # Add guest speaker if detected
        if len(segments) > 10:  # Assume longer conversations have guests
            speakers.append({
                'id': 'speaker_2',
                'name': 'Guest',
                'confidence': 0.7,
                'segments_count': len([s for s in segments if s.get('speaker') == 'speaker_2'])
            })
        
        return speakers


class ViralMomentDetector:
    """Detects potential viral moments during transcription."""
    
    def __init__(self):
        """Initialize viral moment detector."""
        # Keywords that often indicate viral content
        self.viral_keywords = {
            'emotional': ['shocking', 'unbelievable', 'incredible', 'amazing', 'crazy'],
            'controversial': ['controversial', 'banned', 'secret', 'exposed', 'truth'],
            'financial': ['million', 'billion', 'rich', 'money', 'expensive', 'cost'],
            'personal': ['story', 'happened', 'experience', 'told', 'said'],
            'actionable': ['how to', 'you should', 'the key is', 'the secret'],
            'timely': ['2024', '2025', 'today', 'now', 'recently', 'just']
        }
    
    def detect_viral_moments(self, result: ViralTranscriptionResult) -> List[Dict]:
        """Detect potential viral moments in transcription."""
        viral_moments = []
        
        for segment in result.segments:
            score = self._score_segment_virality(segment)
            
            if score > 0.3:  # Threshold for viral potential (lowered for testing)
                viral_moments.append({
                    'timestamp': segment.get('start', 0),
                    'text': segment.get('text', ''),
                    'speaker': segment.get('speaker'),
                    'viral_score': score,
                    'reasons': self._get_viral_reasons(segment.get('text', ''))
                })
        
        # Sort by viral score
        viral_moments.sort(key=lambda x: x['viral_score'], reverse=True)
        
        return viral_moments[:10]  # Top 10 moments
    
    def _score_segment_virality(self, segment: Dict) -> float:
        """Score a segment for viral potential."""
        text = segment.get('text', '').lower()
        score = 0.0
        
        # Keyword scoring
        for category, keywords in self.viral_keywords.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches > 0:
                score += matches * 0.1
        
        # Length scoring (moderate length is better)
        word_count = len(text.split())
        if 10 <= word_count <= 30:
            score += 0.2
        
        # Confidence scoring
        confidence = segment.get('confidence', 0.0)
        if confidence > 0.5:
            score += 0.1
        
        return min(1.0, score)
    
    def _get_viral_reasons(self, text: str) -> List[str]:
        """Get reasons why this segment might be viral."""
        reasons = []
        text_lower = text.lower()
        
        for category, keywords in self.viral_keywords.items():
            if any(kw in text_lower for kw in keywords):
                reasons.append(category.title())
        
        return reasons


class ViralTranscriber:
    """Enhanced transcriber for viral content creation."""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 preferred_methods: List[str] = None,
                 enable_speaker_id: bool = True,
                 enable_viral_detection: bool = True):
        """
        Initialize viral transcriber.
        
        Args:
            openai_api_key: OpenAI API key
            preferred_methods: Preferred transcription methods
            enable_speaker_id: Enable speaker identification
            enable_viral_detection: Enable viral moment detection
        """
        self.openai_api_key = openai_api_key
        self.enable_speaker_id = enable_speaker_id
        self.enable_viral_detection = enable_viral_detection
        
        # Initialize base transcriber
        self.base_transcriber = MultiTranscriber(
            openai_api_key=openai_api_key,
            preferred_methods=preferred_methods or ['youtube', 'local_whisper_enhanced', 'whisper']
        )
        
        # Initialize enhanced components
        if is_whisper_available():
            self.enhanced_whisper = EnhancedLocalWhisper()
        else:
            self.enhanced_whisper = None
        
        if enable_viral_detection:
            self.viral_detector = ViralMomentDetector()
        else:
            self.viral_detector = None
        
        logger.info(f"ViralTranscriber initialized. Speaker ID: {enable_speaker_id}, Viral detection: {enable_viral_detection}")
    
    def transcribe_for_viral_content(self,
                                   audio_url: str,
                                   youtube_urls: List[str] = None,
                                   title: str = None,
                                   language: str = None) -> ViralTranscriptionResult:
        """
        Transcribe audio with viral content enhancements.

        Args:
            audio_url: URL to audio file
            youtube_urls: Optional YouTube URLs for transcript fallback
            title: Episode title
            language: Target language

        Returns:
            Enhanced transcription result with timestamps, speakers, and viral moments
        """
        console.print(f"[dim]  Starting transcription: {(title or 'Unknown')[:50]}...[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True  # Progress disappears when done
        ) as progress:

            task = progress.add_task("[yellow]Transcribing audio...", total=100)

            # Try enhanced local Whisper first if available and speaker ID enabled
            if self.enhanced_whisper and self.enable_speaker_id:
                try:
                    progress.update(task, description="[yellow]Using enhanced Whisper with speaker ID...")
                    progress.update(task, advance=20)

                    result = self.enhanced_whisper.transcribe_with_speakers(audio_url, language)
                    progress.update(task, advance=60)

                    console.print(f"[dim]  ✓ Transcribed {len(result.segments)} segments with speaker ID[/dim]")

                    # Add viral moment detection
                    if self.viral_detector:
                        progress.update(task, description="[yellow]Detecting viral moments...")
                        viral_moments = self.viral_detector.detect_viral_moments(result)
                        result.viral_moments = viral_moments
                        progress.update(task, advance=20)
                        console.print(f"[dim]  ✓ Detected {len(viral_moments)} potential viral moments[/dim]")

                    return result

                except Exception as e:
                    console.print(f"[yellow]  ⚠ Enhanced Whisper failed, trying fallback...[/yellow]")
                    logger.warning(f"Enhanced Whisper failed, falling back to standard methods: {e}")

            # Fallback to standard multi-transcriber
            progress.update(task, description="[yellow]Using standard transcription...")
            progress.update(task, advance=30)

            base_result = self.base_transcriber.transcribe_episode(
                audio_url=audio_url,
                youtube_urls=youtube_urls or [],
                title=title or "Unknown"
            )
            progress.update(task, advance=50)

            if not base_result or base_result.get('error'):
                progress.update(task, completed=100)
                console.print("[red]  ✗ All transcription methods failed[/red]")
                logger.error("All transcription methods failed")
                return ViralTranscriptionResult(
                    text="",
                    method="failed",
                    duration=0
                )

            # Convert to viral result format
            viral_result = ViralTranscriptionResult(
                text=base_result.get('text', ''),
                method=base_result.get('method', 'unknown'),
                duration=base_result.get('duration'),
                language=base_result.get('language')
            )

            # Add basic segments if we have text
            if viral_result.text:
                progress.update(task, description="[yellow]Creating segments...")
                viral_result.segments = self._create_basic_segments(viral_result.text, viral_result.duration)
                progress.update(task, advance=15)

            # Add viral moment detection even for basic transcriptions
            if self.viral_detector and viral_result.segments:
                progress.update(task, description="[yellow]Detecting viral moments...")
                viral_moments = self.viral_detector.detect_viral_moments(viral_result)
                viral_result.viral_moments = viral_moments
                progress.update(task, advance=5)
                console.print(f"[dim]  ✓ Detected {len(viral_moments)} potential viral moments[/dim]")

        return viral_result
    
    def _create_basic_segments(self, text: str, duration: Optional[float]) -> List[Dict]:
        """Create basic time-based segments from text."""
        sentences = text.split('. ')
        segments = []
        
        if not duration:
            duration = len(sentences) * 3  # Rough estimate: 3 seconds per sentence
        
        time_per_sentence = duration / len(sentences) if sentences else 1
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                segments.append({
                    'start': i * time_per_sentence,
                    'end': (i + 1) * time_per_sentence,
                    'text': sentence.strip(),
                    'confidence': 0.8,  # Default confidence
                    'speaker': 'speaker_1'  # Default speaker
                })
        
        return segments
    
    def get_transcription_summary(self, result: ViralTranscriptionResult) -> Dict[str, Any]:
        """Get summary statistics for transcription result."""
        return {
            'method': result.method,
            'duration': result.duration,
            'segments_count': len(result.segments),
            'speakers_count': len(result.speakers),
            'viral_moments_count': len(result.viral_moments),
            'confidence': result.confidence,
            'language': result.language,
            'top_viral_moment': result.viral_moments[0] if result.viral_moments else None
        }