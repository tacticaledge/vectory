"""
Pydantic models and Enums for LLM Evaluation UI.
Single source of truth for data structures.
"""

from enum import Enum, auto
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


# ============================================================================
# Enums
# ============================================================================

class ExtractionMode(str, Enum):
    """PDF/Document extraction modes."""
    MARKDOWN = "markdown"
    TEXT = "text"
    CHUNKS = "chunks"


class DataSourceType(str, Enum):
    """Type of data source loaded."""
    TABULAR = "tabular"      # CSV/JSON with columns (requires column mapping)
    DOCUMENT = "document"    # PDF text content (evaluate document content)
    IMAGE = "image"          # Image files (requires vision model)


class ContentType(str, Enum):
    """Types of extracted content."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    PAGE = "page"


class EvalTaskType(str, Enum):
    """Evaluation task types (aligned with MTEB)."""
    CLASSIFICATION = "Classification"
    CLUSTERING = "Clustering"
    PAIR_CLASSIFICATION = "PairClassification"
    RERANKING = "Reranking"
    RETRIEVAL = "Retrieval"
    STS = "STS"
    SUMMARIZATION = "Summarization"
    BITEXT_MINING = "BitextMining"


class RatingType(str, Enum):
    """Human evaluation rating types."""
    SCALE_1_5 = "1-5 Scale"
    THUMBS = "Thumbs Up/Down"
    PASS_FAIL = "Pass/Fail"


class MetricType(str, Enum):
    """Rule-based evaluation metrics."""
    BLEU = "bleu"
    ROUGE_1 = "rouge1"
    ROUGE_2 = "rouge2"
    ROUGE_L = "rougeL"
    EXACT_MATCH = "exact_match"
    LEVENSHTEIN = "levenshtein"
    REGEX = "regex"


# ============================================================================
# Document Models
# ============================================================================

class ExtractedContent(BaseModel):
    """A piece of content extracted from a document."""
    model_config = ConfigDict(extra="allow")

    content_type: ContentType
    text: str
    source_file: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentResult(BaseModel):
    """Result of processing a document."""
    filename: str
    content: List[ExtractedContent]
    page_count: int = 0
    extraction_mode: ExtractionMode
    error: Optional[str] = None

    @property
    def text(self) -> str:
        """Concatenate all text content."""
        return "\n\n".join(c.text for c in self.content if c.text)


# ============================================================================
# Dataset Models
# ============================================================================

class ColumnMapping(BaseModel):
    """Mapping of dataset columns to semantic roles."""
    input_col: Optional[str] = Field(None, description="Input/prompt column")
    output_col: Optional[str] = Field(None, description="LLM output column")
    expected_col: Optional[str] = Field(None, description="Expected/reference column")
    label_col: Optional[str] = Field(None, description="Label/category column")

    def is_valid(self) -> bool:
        """Check if minimum required mapping exists."""
        return self.output_col is not None


class DatasetInfo(BaseModel):
    """Information about a loaded dataset."""
    filename: str
    row_count: int
    columns: List[str]
    column_mapping: Optional[ColumnMapping] = None


# ============================================================================
# Evaluation Models
# ============================================================================

class EvaluationResult(BaseModel):
    """Result of evaluating a single sample."""
    index: int
    score: float
    metrics: Dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None


class HumanAnnotation(BaseModel):
    """A human evaluation annotation."""
    sample_index: int
    rating: float
    rating_type: RatingType
    criteria: List[str] = Field(default_factory=list)
    feedback: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class MetricResult(BaseModel):
    """Result of a metric evaluation."""
    metric: MetricType
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Embedding/Benchmark Models
# ============================================================================

class EmbeddingModel(BaseModel):
    """Configuration for an embedding model."""
    name: str
    provider: str
    model_id: str
    dimensions: int
    max_tokens: int

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.provider})"


class BenchmarkResult(BaseModel):
    """Result of benchmarking an embedding model."""
    model: EmbeddingModel
    task_scores: Dict[EvalTaskType, float]
    mean_score: float = 0.0
    rank: int = 0

    def model_post_init(self, __context):
        if self.task_scores and self.mean_score == 0.0:
            self.mean_score = sum(self.task_scores.values()) / len(self.task_scores)


# ============================================================================
# Session State Model (for Streamlit)
# ============================================================================

class AppState(BaseModel):
    """Application state for Streamlit session."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Optional[Any] = None  # DataFrame
    dataset_info: Optional[DatasetInfo] = None
    column_mapping: ColumnMapping = Field(default_factory=ColumnMapping)
    evaluation_results: Dict[str, Any] = Field(default_factory=dict)
    human_annotations: Dict[int, HumanAnnotation] = Field(default_factory=dict)
    benchmark_results: Optional[Any] = None  # DataFrame


def init_session_state(st) -> None:
    """Initialize Streamlit session state with proper defaults."""
    defaults = {
        "dataset": None,
        "dataset_filename": None,
        "dataset_info": None,
        "data_source_type": None,  # DataSourceType: tabular, document, or image
        "column_mapping": ColumnMapping(),
        "evaluation_results": {},
        "human_annotations": {},
        "benchmark_results": None,
        "current_sample_idx": 0,
        # For document/image evaluation
        "raw_file_content": None,  # Store raw bytes for image evaluation
        # Theme settings
        "theme": "dark",  # Default theme: dark, light, colorblind, high_contrast
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
