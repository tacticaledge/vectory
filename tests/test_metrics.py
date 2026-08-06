"""
Tests for metrics evaluators.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.evaluators.metrics import (
    ExactMatchEvaluator,
    ContainsEvaluator,
    LevenshteinEvaluator,
    BLEUEvaluator,
    ROUGEEvaluator,
    RegexEvaluator,
)


class TestExactMatchEvaluator:
    """Tests for ExactMatchEvaluator."""

    def test_exact_match_true(self):
        """Test exact match returns 1.0 for identical strings."""
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate_single("Hello World", "Hello World")
        assert result["exact_match"] == 1.0

    def test_exact_match_false(self):
        """Test exact match returns 0.0 for different strings."""
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate_single("Hello World", "Goodbye World")
        assert result["exact_match"] == 0.0

    def test_exact_match_case_insensitive(self):
        """Test case-insensitive matching (default)."""
        evaluator = ExactMatchEvaluator(case_sensitive=False)
        result = evaluator.evaluate_single("HELLO", "hello")
        assert result["exact_match"] == 1.0

    def test_exact_match_case_sensitive(self):
        """Test case-sensitive matching."""
        evaluator = ExactMatchEvaluator(case_sensitive=True)
        result = evaluator.evaluate_single("HELLO", "hello")
        assert result["exact_match"] == 0.0

    def test_exact_match_whitespace_strip(self):
        """Test whitespace stripping (default)."""
        evaluator = ExactMatchEvaluator(strip_whitespace=True)
        result = evaluator.evaluate_single("  hello  ", "hello")
        assert result["exact_match"] == 1.0

    def test_exact_match_no_reference(self):
        """Test error handling when no reference provided."""
        evaluator = ExactMatchEvaluator()
        result = evaluator.evaluate_single("Hello", None)
        assert result["exact_match"] is None
        assert "error" in result


class TestContainsEvaluator:
    """Tests for ContainsEvaluator."""

    def test_contains_true(self):
        """Test contains returns 1.0 when reference is in output."""
        evaluator = ContainsEvaluator()
        result = evaluator.evaluate_single("The answer is 42", "42")
        assert result["contains"] == 1.0

    def test_contains_false(self):
        """Test contains returns 0.0 when reference not in output."""
        evaluator = ContainsEvaluator()
        result = evaluator.evaluate_single("The answer is 42", "43")
        assert result["contains"] == 0.0


class TestLevenshteinEvaluator:
    """Tests for LevenshteinEvaluator."""

    def test_levenshtein_identical(self):
        """Test identical strings have similarity 1.0."""
        evaluator = LevenshteinEvaluator()
        result = evaluator.evaluate_single("hello", "hello")
        assert result["levenshtein_similarity"] == 1.0
        assert result["levenshtein_distance"] == 0

    def test_levenshtein_different(self):
        """Test different strings have similarity < 1.0."""
        evaluator = LevenshteinEvaluator()
        result = evaluator.evaluate_single("hello", "hallo")
        assert 0 < result["levenshtein_similarity"] < 1.0
        assert result["levenshtein_distance"] == 1

    def test_levenshtein_completely_different(self):
        """Test completely different strings."""
        evaluator = LevenshteinEvaluator()
        result = evaluator.evaluate_single("abc", "xyz")
        assert result["levenshtein_similarity"] < 0.5


class TestBLEUEvaluator:
    """Tests for BLEUEvaluator."""

    def test_bleu_identical(self):
        """Test BLEU score for identical strings."""
        evaluator = BLEUEvaluator()
        result = evaluator.evaluate_single(
            "The quick brown fox",
            "The quick brown fox"
        )
        assert result["bleu"] is not None
        # Identical strings should have high BLEU
        assert result["bleu"] > 0.5

    def test_bleu_different(self):
        """Test BLEU score for different strings."""
        evaluator = BLEUEvaluator()
        result = evaluator.evaluate_single(
            "The lazy dog",
            "A quick fox jumps"
        )
        assert result["bleu"] is not None
        # Different strings should have low BLEU
        assert result["bleu"] < 0.5

    def test_bleu_no_reference(self):
        """Test error handling when no reference."""
        evaluator = BLEUEvaluator()
        result = evaluator.evaluate_single("Hello", None)
        assert result["bleu"] is None
        assert "error" in result


class TestROUGEEvaluator:
    """Tests for ROUGEEvaluator."""

    def test_rouge_scores(self):
        """Test ROUGE scores are computed."""
        evaluator = ROUGEEvaluator()
        result = evaluator.evaluate_single(
            "The cat sat on the mat",
            "The cat is on the mat"
        )
        # Should have ROUGE-1, ROUGE-2, ROUGE-L scores
        assert any("rouge" in k.lower() for k in result.keys())

    def test_rouge_identical(self):
        """Test ROUGE for identical strings."""
        evaluator = ROUGEEvaluator()
        result = evaluator.evaluate_single(
            "Hello world",
            "Hello world"
        )
        # At least one ROUGE score should be high for identical text
        rouge_scores = [v for k, v in result.items() if "rouge" in k.lower() and isinstance(v, (int, float))]
        if rouge_scores:
            assert max(rouge_scores) > 0.5


class TestRegexEvaluator:
    """Tests for RegexEvaluator."""

    def test_regex_match(self):
        """Test regex matching."""
        evaluator = RegexEvaluator(r"\d+")  # Match digits
        result = evaluator.evaluate_single("The answer is 42")
        assert result["regex_match"] == 1.0

    def test_regex_no_match(self):
        """Test regex no match."""
        evaluator = RegexEvaluator(r"\d+")  # Match digits
        result = evaluator.evaluate_single("No numbers here")
        assert result["regex_match"] == 0.0

    def test_regex_email_pattern(self):
        """Test email pattern matching."""
        evaluator = RegexEvaluator(r"[\w.-]+@[\w.-]+\.\w+")
        result = evaluator.evaluate_single("Contact: test@example.com")
        assert result["regex_match"] == 1.0
