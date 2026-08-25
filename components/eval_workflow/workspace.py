"""Portable, versioned review workspaces for local and CI-friendly workflows."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, default=str))
            handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def build_review_bundle(
    records: Iterable[Mapping[str, Any]],
    samples: Iterable[Mapping[str, Any]],
    annotations: Mapping[int | str, Mapping[str, Any]] | None = None,
    taxonomy: Mapping[str, Mapping[str, Any]] | None = None,
    suggestions: Mapping[str, Any] | list[Any] | None = None,
    evaluators: Mapping[str, Mapping[str, Any]] | None = None,
    source: str | None = None,
    reviewer: Mapping[str, Any] | None = None,
    dimension_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a self-contained, versioned review artifact."""
    record_list = [dict(record) for record in records]
    annotation_map = {str(key): dict(value) for key, value in (annotations or {}).items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source": source,
        "review_protocol": {
            "verdict": "binary_pass_fail",
            "critique_required": True,
            "principal_domain_expert": dict(reviewer or {}),
            "dimension_fields": list(dimension_fields or []),
            "discovery_stop_condition": (
                "Continue expert review until new failure modes stop appearing; start around 30 examples "
                "and expand when the domain remains heterogeneous."
            ),
            "lifecycle": [
                "discover",
                "expert_review",
                "fix_obvious_errors",
                "promote_evaluator",
                "validate_on_held_out_labels",
                "monitor_and_repeat",
            ],
        },
        "dataset": {
            "record_count": len(record_list),
            "sha256": hashlib.sha256(_canonical_json(record_list).encode("utf-8")).hexdigest(),
            "records": record_list,
        },
        "sampling": {
            "purpose": "failure_mode_discovery",
            "unbiased_prevalence_estimate": False,
            "warning": (
                "Cluster representatives and targeted suggestions are biased toward coverage. "
                "Use a separate random sample for prevalence estimates."
            ),
            "samples": [dict(sample) for sample in samples],
        },
        "annotations": annotation_map,
        "taxonomy": dict(taxonomy or {}),
        "suggestions": suggestions or {},
        "evaluators": {str(key): dict(value) for key, value in (evaluators or {}).items()},
    }


def save_review_bundle(path: str | Path, bundle: Mapping[str, Any]) -> Path:
    target = Path(path)
    _atomic_write_json(target, dict(bundle))
    return target


def save_json_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a generic Vectory JSON artifact."""
    target = Path(path)
    _atomic_write_json(target, dict(payload))
    return target


def save_jsonl_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Atomically write JSONL records."""
    target = Path(path)
    _write_jsonl(target, records)
    return target


def save_evaluator_definition(
    workspace: str | Path, evaluator: Mapping[str, Any]
) -> Path:
    """Persist an evaluator definition inside a review workspace."""
    from .promotion import validate_evaluator_definition

    evaluator = validate_evaluator_definition(evaluator)
    evaluator_id = str(evaluator.get("evaluator_id") or "").strip()
    if not evaluator_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in evaluator_id):
        raise ValueError("Evaluator definition must have a safe evaluator_id")
    root = Path(workspace)
    if not (root / "manifest.json").is_file():
        raise ValueError(f"Not a Vectory review workspace: {root}")
    target = root / "evaluators" / f"{evaluator_id}.json"
    _atomic_write_json(target, dict(evaluator))
    return target


def archive_review_workspace(path: str | Path) -> Path:
    """Atomically archive an existing workspace before a discovery refresh."""
    root = Path(path)
    if not (root / "manifest.json").is_file():
        raise ValueError(f"Not a Vectory review workspace: {root}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = root.with_name(f"{root.name}.archive-{timestamp}")
    root.replace(archive)
    return archive


def initialize_review_workspace(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    samples: Iterable[Mapping[str, Any]],
    source: str | None = None,
    reviewer: Mapping[str, Any] | None = None,
    dimension_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Create a directory workspace with inspectable, independently editable artifacts."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    bundle = build_review_bundle(
        records,
        samples,
        source=source,
        reviewer=reviewer,
        dimension_fields=dimension_fields,
    )
    manifest = {
        key: bundle[key]
        for key in ("schema_version", "created_at", "source", "review_protocol")
    }
    manifest["dataset"] = {
        "record_count": bundle["dataset"]["record_count"],
        "sha256": bundle["dataset"]["sha256"],
        "path": "records.jsonl",
    }
    manifest["sampling"] = {
        key: bundle["sampling"][key]
        for key in ("purpose", "unbiased_prevalence_estimate", "warning")
    }
    manifest["artifacts"] = {
        "samples": "samples.json",
        "annotations": "annotations.json",
        "taxonomy": "taxonomy.json",
        "suggestions": "suggestions.json",
        "evaluators": "evaluators",
    }

    _atomic_write_json(root / "manifest.json", manifest)
    _write_jsonl(root / "records.jsonl", bundle["dataset"]["records"])
    _atomic_write_json(root / "samples.json", bundle["sampling"]["samples"])
    _atomic_write_json(root / "annotations.json", bundle["annotations"])
    _atomic_write_json(root / "taxonomy.json", bundle["taxonomy"])
    _atomic_write_json(root / "suggestions.json", bundle["suggestions"])
    (root / "evaluators").mkdir(exist_ok=True)
    return manifest


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def load_review_workspace(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = _read_json(root / "manifest.json", None)
    if not isinstance(manifest, dict):
        raise ValueError(f"Not a Vectory review workspace: {root}")
    records_path = root / manifest.get("dataset", {}).get("path", "records.jsonl")
    evaluator_dir = root / "evaluators"
    evaluators = {}
    if evaluator_dir.is_dir():
        for evaluator_path in sorted(evaluator_dir.glob("*.json")):
            evaluator = _read_json(evaluator_path, {})
            evaluator_id = evaluator.get("evaluator_id", evaluator_path.stem)
            evaluators[str(evaluator_id)] = evaluator
    return {
        "root": root,
        "manifest": manifest,
        "records": _read_jsonl(records_path),
        "samples": _read_json(root / "samples.json", []),
        "annotations": _read_json(root / "annotations.json", {}),
        "taxonomy": _read_json(root / "taxonomy.json", {}),
        "suggestions": _read_json(root / "suggestions.json", {}),
        "evaluators": evaluators,
    }
