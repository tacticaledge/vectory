"""Suite loading utilities for VectoryBenchmark."""

from __future__ import annotations

import json
from pathlib import Path

from components.vectory_benchmark.schemas import BenchmarkSuite, BenchmarkTask


DEFAULT_SUITE_PATH = Path(__file__).parent.parent.parent / "data" / "vectory_benchmark" / "manifest.json"


def load_suite(path: str | Path | None = None) -> BenchmarkSuite:
    """Load a VectoryBenchmark suite manifest."""
    suite_path = Path(path) if path else DEFAULT_SUITE_PATH
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    return BenchmarkSuite(**payload)


def get_task_by_id(task_id: str, path: str | Path | None = None) -> BenchmarkTask:
    """Load one task by ID."""
    suite = load_suite(path)
    for task in suite.tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")
