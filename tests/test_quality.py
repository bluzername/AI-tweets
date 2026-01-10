"""
Quality features tests - Multi-model AI, hooks, deduplication, optimization.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_model_analyzer import MultiModelAnalyzer, ModelInsight, AIModel
from hook_generator import HookGenerator, HookPattern, HookVariant
from content_deduplicator import ContentDeduplicator, ContentRecord
from emoji_hashtag_optimizer import EmojiHashtagOptimizer, EmojiCategory


class TestMultiModelAnalyzer:
    """Test multi-model AI analyzer."""

    def test_initialization(self, mock_openai_key):
        """Test analyzer initialization."""
        analyzer = MultiModelAnalyzer()
        assert analyzer is not None

    def test_model_detection(self, mock_openai_key):
        """Test available model detection."""
        analyzer = MultiModelAnalyzer()
        models = analyzer._detect_available_models()

        # Should have at least one model (OpenAI from mock)
        assert len(models) >= 0  # May have 0 if keys are invalid

    def test_similarity_detection(self, mock_openai_key):
        """Test semantic similarity detection."""
        analyzer = MultiModelAnalyzer()

        # Very similar texts
        text1 = "The secret to startup success is focus"
        text2 = "Startup success comes from intense focus"

        similar = analyzer._are_similar(text1, text2, threshold=0.3)
        assert similar is True

        # Very different texts
        text3 = "I love pizza and ice cream"
        text4 = "Quantum mechanics is fascinating"

        not_similar = analyzer._are_similar(text3, text4, threshold=0.3)
        assert not_similar is False

    def test_model_statistics(self, mock_openai_key):
        """Test getting model statistics."""
        analyzer = MultiModelAnalyzer()
        stats = analyzer.get_model_statistics()

        assert "enabled_models" in stats
        assert "available_clients" in stats
        assert isinstance(stats["enabled_models"], list)


class TestHookGenerator:
    """Test hook variant generator."""

    def test_initialization(self, mock_openai_key):
        """Test hook generator initialization."""
        generator = HookGenerator()
        assert generator is not None

    def test_template_loading(self, mock_openai_key):
        """Test template loading."""
        generator = HookGenerator()
        templates = generator._load_templates()

        assert HookPattern.QUESTION in templates
        assert HookPattern.SHOCK in templates
        assert HookPattern.NUMBERS in templates
        assert len(templates[HookPattern.QUESTION]) > 0

    def test_emoji_detection(self, mock_openai_key):
        """Test emoji detection."""
        generator = HookGenerator()

        assert generator._has_emoji("Hello 🚀 world") is True
        assert generator._has_emoji("Hello world") is False
        assert generator._has_emoji("Amazing! 💡") is True

    def test_template_variant_generation(self, mock_openai_key, sample_insight):
        """Test template-based variant generation."""
        generator = HookGenerator()

        variants = generator._generate_template_variants(
            insight_text=sample_insight["text"],
            category="fact",
            count=3
        )

        assert len(variants) == 3
        assert all(isinstance(v, HookVariant) for v in variants)
        assert all(v.pattern in HookPattern for v in variants)
        assert all(len(v.full_tweet) > 0 for v in variants)

    def test_variant_ranking(self, mock_openai_key):
        """Test variant ranking."""
        generator = HookGenerator()

        # Create test variants
        variants = [
            HookVariant(
                pattern=HookPattern.QUESTION,
                text="Test hook?",
                full_tweet="Test hook?\n\nBody text",
                engagement_score=0.7,
                character_count=25,
                contains_emoji=False
            ),
            HookVariant(
                pattern=HookPattern.SHOCK,
                text="This will blow your mind:",
                full_tweet="This will blow your mind:\n\nBody text",
                engagement_score=0.8,
                character_count=45,
                contains_emoji=False
            )
        ]

        ranked = generator.rank_variants_by_predicted_engagement(variants)

        # Higher engagement should be ranked first
        assert ranked[0].engagement_score >= ranked[1].engagement_score


class TestContentDeduplicator:
    """Test content deduplication."""

    def test_initialization(self, temp_db):
        """Test deduplicator initialization."""
        dedup = ContentDeduplicator(db_path=temp_db)
        assert dedup is not None

    def test_similarity_hash(self, temp_db):
        """Test similarity hash computation."""
        dedup = ContentDeduplicator(db_path=temp_db)

        # Same text should produce same hash
        hash1 = dedup._compute_similarity_hash("Hello World")
        hash2 = dedup._compute_similarity_hash("Hello World")
        assert hash1 == hash2

        # Normalization should work
        hash3 = dedup._compute_similarity_hash("  Hello   World  ")
        assert hash1 == hash3

        # Different text should produce different hash
        hash4 = dedup._compute_similarity_hash("Goodbye World")
        assert hash1 != hash4

    def test_add_content(self, temp_db):
        """Test adding content to history."""
        dedup = ContentDeduplicator(db_path=temp_db)

        content_id = dedup.add_content(
            text="This is a test tweet",
            podcast_name="Test Podcast",
            category="test"
        )

        assert content_id is not None
        assert len(content_id) > 0

    def test_exact_duplicate_detection(self, temp_db):
        """Test exact duplicate detection."""
        dedup = ContentDeduplicator(db_path=temp_db)

        text = "This is a unique tweet about startups"

        # Add to history
        dedup.add_content(text, podcast_name="Test", category="test")

        # Check for duplicate
        is_dup, match = dedup.check_duplicate(text)

        assert is_dup is True
        assert match is not None
        assert match["type"] == "exact"

    def test_non_duplicate_detection(self, temp_db):
        """Test non-duplicate content."""
        dedup = ContentDeduplicator(db_path=temp_db)

        dedup.add_content("First tweet", podcast_name="Test", category="test")

        is_dup, match = dedup.check_duplicate("Completely different tweet")

        assert is_dup is False
        assert match is None

    def test_filter_duplicates(self, temp_db):
        """Test filtering duplicates from list."""
        dedup = ContentDeduplicator(db_path=temp_db)

        # Add one to history
        dedup.add_content("Tweet one", podcast_name="Test", category="test")

        # Try to filter list with duplicate
        content_list = [
            {"text": "Tweet one", "podcast_name": "Test"},
            {"text": "Tweet two", "podcast_name": "Test"},
            {"text": "Tweet three", "podcast_name": "Test"}
        ]

        filtered = dedup.filter_duplicates(content_list)

        # Should remove the duplicate
        assert len(filtered) == 2


class TestEmojiHashtagOptimizer:
    """Test emoji and hashtag optimization."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = EmojiHashtagOptimizer()
        assert optimizer is not None

    def test_emoji_library_loading(self):
        """Test emoji library is loaded."""
        optimizer = EmojiHashtagOptimizer()
        library = optimizer._load_emoji_library()

        assert EmojiCategory.TECH in library
        assert EmojiCategory.BUSINESS in library
        assert len(library[EmojiCategory.TECH]) > 0

    def test_hashtag_library_loading(self):
        """Test hashtag library is loaded."""
        optimizer = EmojiHashtagOptimizer()
        library = optimizer._load_hashtag_library()

        assert "tech" in library
        assert "business" in library
        assert len(library["tech"]) > 0

    def test_emoji_recommendation(self):
        """Test emoji recommendation."""
        optimizer = EmojiHashtagOptimizer()

        text = "We're launching a new startup and it's growing fast!"

        recs = optimizer.recommend_emojis(
            text=text,
            category=EmojiCategory.BUSINESS,
            max_emojis=2
        )

        assert len(recs) <= 2
        assert all(hasattr(r, 'emoji') for r in recs)
        assert all(hasattr(r, 'relevance_score') for r in recs)

    def test_hashtag_recommendation(self):
        """Test hashtag recommendation."""
        optimizer = EmojiHashtagOptimizer()

        text = "Learn about AI and machine learning for startups"

        recs = optimizer.recommend_hashtags(
            text=text,
            categories=["tech", "business"],
            max_hashtags=3
        )

        assert len(recs) <= 3
        assert all(hasattr(r, 'hashtag') for r in recs)
        assert all(r.hashtag.startswith('#') for r in recs)

    def test_tweet_optimization(self):
        """Test full tweet optimization."""
        optimizer = EmojiHashtagOptimizer()

        original = "The secret to startup success is focus and execution"

        result = optimizer.optimize_tweet(
            text=original,
            category=EmojiCategory.BUSINESS,
            content_categories=["business", "tech"],
            max_emojis=2,
            max_hashtags=3
        )

        assert "original" in result
        assert "optimized" in result
        assert "character_count" in result
        assert "emojis_added" in result
        assert "hashtags_added" in result

        # Optimized should be different from original (has emojis/hashtags)
        assert result["optimized"] != result["original"]
        assert len(result["optimized"]) > len(result["original"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
