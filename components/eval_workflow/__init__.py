"""Reusable primitives for Vectory's human-in-the-loop evaluation workflow."""

from .discovery import (
    dataframe_to_records,
    find_related_records,
    select_diverse_samples,
    summarize_coverage,
)
from .promotion import build_evaluator_definition, validate_evaluator_definition
from .trace_review import normalize_trace_segments
from .validation import (
    bootstrap_metric_intervals,
    normalize_binary_label,
    split_labeled_records,
    validate_evaluator_labels,
)
from .workspace import (
    archive_review_workspace,
    build_review_bundle,
    initialize_review_workspace,
    load_review_workspace,
    save_evaluator_definition,
    save_json_artifact,
    save_jsonl_records,
    save_review_bundle,
)

__all__ = [
    "archive_review_workspace",
    "bootstrap_metric_intervals",
    "build_evaluator_definition",
    "build_review_bundle",
    "dataframe_to_records",
    "find_related_records",
    "initialize_review_workspace",
    "load_review_workspace",
    "normalize_binary_label",
    "normalize_trace_segments",
    "save_evaluator_definition",
    "save_json_artifact",
    "save_jsonl_records",
    "save_review_bundle",
    "select_diverse_samples",
    "split_labeled_records",
    "summarize_coverage",
    "validate_evaluator_labels",
    "validate_evaluator_definition",
]
