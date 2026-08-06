"""
Tests for Pydantic data models.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from components.models import (
    ColumnMapping,
    ExtractionMode,
    ContentType,
    EvalTaskType,
    RatingType,
    MetricType,
    EmbeddingModel,
    ExtractedContent,
    DocumentResult,
)


class TestColumnMapping:
    """Tests for ColumnMapping model."""

    def test_column_mapping_valid(self):
        """Test valid column mapping with output_col set."""
        mapping = ColumnMapping(output_col="response")
        assert mapping.is_valid() is True
        assert mapping.output_col == "response"

    def test_column_mapping_invalid(self):
        """Test invalid column mapping without output_col."""
        mapping = ColumnMapping()
        assert mapping.is_valid() is False

    def test_column_mapping_full(self):
        """Test column mapping with all fields."""
        mapping = ColumnMapping(
            input_col="question",
            output_col="answer",
            expected_col="reference",
            label_col="category"
        )
        assert mapping.is_valid() is True
        assert mapping.input_col == "question"
        assert mapping.output_col == "answer"
        assert mapping.expected_col == "reference"
        assert mapping.label_col == "category"


class TestExtractionMode:
    """Tests for ExtractionMode enum."""

    def test_extraction_mode_values(self):
        """Test extraction mode enum values."""
        assert ExtractionMode.MARKDOWN.value == "markdown"
        assert ExtractionMode.TEXT.value == "text"
        assert ExtractionMode.CHUNKS.value == "chunks"

    def test_extraction_mode_from_string(self):
        """Test creating extraction mode from string."""
        mode = ExtractionMode("markdown")
        assert mode == ExtractionMode.MARKDOWN


class TestContentType:
    """Tests for ContentType enum."""

    def test_content_type_values(self):
        """Test content type enum values."""
        assert ContentType.TEXT.value == "text"
        assert ContentType.TABLE.value == "table"
        assert ContentType.IMAGE.value == "image"
        assert ContentType.PAGE.value == "page"


class TestEvalTaskType:
    """Tests for EvalTaskType enum."""

    def test_eval_task_type_values(self):
        """Test evaluation task type enum values."""
        assert EvalTaskType.CLASSIFICATION.value == "Classification"
        assert EvalTaskType.STS.value == "STS"
        assert EvalTaskType.RETRIEVAL.value == "Retrieval"


class TestRatingType:
    """Tests for RatingType enum."""

    def test_rating_type_values(self):
        """Test rating type enum values."""
        assert RatingType.SCALE_1_5.value == "1-5 Scale"
        assert RatingType.THUMBS.value == "Thumbs Up/Down"
        assert RatingType.PASS_FAIL.value == "Pass/Fail"


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_type_values(self):
        """Test metric type enum values."""
        assert MetricType.BLEU.value == "bleu"
        assert MetricType.ROUGE_1.value == "rouge1"
        assert MetricType.EXACT_MATCH.value == "exact_match"


class TestEmbeddingModel:
    """Tests for EmbeddingModel model."""

    def test_embedding_model_creation(self):
        """Test creating an embedding model."""
        model = EmbeddingModel(
            name="test-model",
            provider="Test Provider",
            model_id="test/model-v1",
            dimensions=768,
            max_tokens=512
        )
        assert model.name == "test-model"
        assert model.provider == "Test Provider"
        assert model.dimensions == 768

    def test_embedding_model_display_name(self):
        """Test embedding model display name property."""
        model = EmbeddingModel(
            name="MiniLM",
            provider="Sentence Transformers",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            max_tokens=256
        )
        assert model.display_name == "MiniLM (Sentence Transformers)"


class TestExtractedContent:
    """Tests for ExtractedContent model."""

    def test_extracted_content_creation(self):
        """Test creating extracted content."""
        content = ExtractedContent(
            content_type=ContentType.TEXT,
            text="Hello world",
            source_file="test.pdf",
            page=1,
            metadata={"format": "markdown"}
        )
        assert content.content_type == ContentType.TEXT
        assert content.text == "Hello world"
        assert content.source_file == "test.pdf"
        assert content.page == 1
        assert content.metadata["format"] == "markdown"


class TestDocumentResult:
    """Tests for DocumentResult model."""

    def test_document_result_creation(self):
        """Test creating a document result."""
        content = ExtractedContent(
            content_type=ContentType.TEXT,
            text="Test content",
            source_file="test.pdf"
        )
        result = DocumentResult(
            filename="test.pdf",
            content=[content],
            page_count=1,
            extraction_mode=ExtractionMode.MARKDOWN
        )
        assert result.filename == "test.pdf"
        assert len(result.content) == 1
        assert result.page_count == 1
        assert result.error is None

    def test_document_result_text_property(self):
        """Test document result text concatenation."""
        contents = [
            ExtractedContent(
                content_type=ContentType.TEXT,
                text="First paragraph",
                source_file="test.pdf"
            ),
            ExtractedContent(
                content_type=ContentType.TEXT,
                text="Second paragraph",
                source_file="test.pdf"
            ),
        ]
        result = DocumentResult(
            filename="test.pdf",
            content=contents,
            page_count=2,
            extraction_mode=ExtractionMode.TEXT
        )
        assert "First paragraph" in result.text
        assert "Second paragraph" in result.text

    def test_document_result_with_error(self):
        """Test document result with error."""
        result = DocumentResult(
            filename="broken.pdf",
            content=[],
            page_count=0,
            extraction_mode=ExtractionMode.MARKDOWN,
            error="Failed to parse PDF"
        )
        assert result.error == "Failed to parse PDF"
        assert len(result.content) == 0
