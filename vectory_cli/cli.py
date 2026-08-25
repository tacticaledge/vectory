"""Vectory command line interface."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from components.eval_workflow import (
    JudgeValidationReport,
    archive_review_workspace,
    bootstrap_metric_intervals,
    build_evaluator_definition,
    dataframe_to_records,
    dump_evaluator_definition,
    initialize_review_workspace,
    load_review_workspace,
    parse_evaluator_definition,
    save_evaluator_definition,
    save_json_artifact,
    save_jsonl_records,
    select_diverse_samples,
    split_labeled_records,
    validate_evaluator_labels,
)

from . import __version__


def _source_app_dir() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    if (root / "app.py").is_file() and (root / "pages").is_dir():
        return root
    return None


def _bundled_app_dir() -> Path | None:
    try:
        candidate = resources.files("vectory_cli").joinpath("streamlit_app")
    except (ModuleNotFoundError, AttributeError):
        return None
    if candidate.joinpath("app.py").is_file():
        return Path(str(candidate))
    return None


def app_dir() -> Path:
    for candidate in (_source_app_dir(), _bundled_app_dir()):
        if candidate is not None:
            return candidate
    raise RuntimeError("Could not locate the Vectory Streamlit application files.")


def _merged_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not existing else f"{root}{os.pathsep}{existing}"
    return env


def run_app(args: argparse.Namespace) -> int:
    root = app_dir()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "app.py"),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
    ]
    if args.headless:
        command.extend(["--server.headless", "true"])
    command.extend(args.streamlit_args)
    return subprocess.call(command, cwd=str(root), env=_merged_env(root))


def run_benchmark(args: argparse.Namespace) -> int:
    root = app_dir()
    command = [
        sys.executable,
        str(root / "scripts" / "run_vectory_benchmark.py"),
        str(args.submission),
    ]
    if args.suite:
        command.extend(["--suite", str(args.suite)])
    if args.scores_out:
        command.extend(["--scores-out", str(args.scores_out)])
    if args.leaderboard_out:
        command.extend(["--leaderboard-out", str(args.leaderboard_out)])
    if args.report_out:
        command.extend(["--report-out", str(args.report_out)])
    if getattr(args, "gate_min_score", None) is not None:
        command.extend(["--gate-min-score", str(args.gate_min_score)])
    if getattr(args, "gate_block_severity", None):
        command.extend(["--gate-block-severity", str(args.gate_block_severity)])
    if getattr(args, "gate_max_pathology_risk", None) is not None:
        command.extend(["--gate-max-pathology-risk", str(args.gate_max_pathology_risk)])
    if args.workspace:
        command.extend(["--workspace", str(args.workspace)])
    if args.allow_formal_runtime:
        command.append("--allow-formal-runtime")
    return subprocess.call(command, cwd=str(root), env=_merged_env(root))


def run_gate(args: argparse.Namespace) -> int:
    root = app_dir()
    command = [
        sys.executable,
        str(root / "scripts" / "run_vectory_benchmark.py"),
        str(args.submission),
        "--gate-min-score",
        str(args.min_score),
        "--gate-block-severity",
        args.block_severity,
    ]
    if args.suite:
        command.extend(["--suite", str(args.suite)])
    if args.max_pathology_risk is not None:
        command.extend(["--gate-max-pathology-risk", str(args.max_pathology_risk)])
    if args.report_out:
        command.extend(["--report-out", str(args.report_out)])
    return subprocess.call(command, cwd=str(root), env=_merged_env(root))


def _load_records(path: Path) -> list[dict]:
    from components.document_loader import load_csv, load_json

    suffix = path.suffix.casefold()
    content = path.read_bytes()
    if suffix in {".json", ".jsonl"}:
        dataframe = load_json(content)
    elif suffix == ".csv":
        dataframe = load_csv(content)
    else:
        raise ValueError("Dataset must be JSON, JSONL, or CSV")
    records = dataframe_to_records(dataframe)
    if not records:
        raise ValueError(f"Dataset contains no records: {path}")
    return records


def _write_json(path: Path, payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError("JSON artifacts must be objects")
    save_json_artifact(path, payload)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    save_jsonl_records(path, records)


def _records_sha256(records: list[dict], fields: list[str] | None = None) -> str:
    projected = records
    if fields is not None:
        projected = [
            {field: record[field] for field in fields if field in record}
            for record in records
        ]
    canonical = json.dumps(
        projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_test_split_provenance(
    dataset: Path,
    records: list[dict],
    manifest_path: Path,
    human_label_column: str,
    judge_label_column: str,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Split manifest must use schema_version 1.0")
    if manifest.get("label_column") != human_label_column:
        raise ValueError(
            "Human label column does not match the trusted label column in the split manifest"
        )
    if manifest.get("judge_label_column") != judge_label_column:
        raise ValueError(
            "Judge label column does not match the prediction field designated by the split manifest"
        )
    test_artifact = (manifest.get("artifacts") or {}).get("test")
    if not isinstance(test_artifact, dict):
        raise TypeError("Split manifest does not contain test artifact provenance")
    fields = test_artifact.get("provenance_fields")
    field_hashes = test_artifact.get("field_sha256")
    expected_count = test_artifact.get("record_count")
    if (
        not isinstance(fields, list)
        or not all(isinstance(field, str) for field in fields)
        or not isinstance(field_hashes, dict)
        or not all(
            isinstance(field, str) and isinstance(digest, str)
            for field, digest in field_hashes.items()
        )
        or not isinstance(expected_count, int)
    ):
        raise ValueError("Split manifest test provenance is incomplete")
    mutable_fields = test_artifact.get("mutable_fields")
    if mutable_fields != [judge_label_column]:
        raise ValueError("Split manifest must designate exactly one mutable judge prediction field")
    verified_fields = [field for field in fields if field not in mutable_fields]
    fields_match = all(
        field_hashes.get(field) == _records_sha256(records, [field])
        for field in verified_fields
    )
    if len(records) != expected_count or not fields_match:
        raise ValueError(
            "Validation dataset does not match the untouched test partition in the split manifest"
        )
    return {
        "verified": True,
        "split_manifest": str(manifest_path),
        "test_artifact": str(dataset),
        "verified_fields": verified_fields,
        "excluded_prediction_field": judge_label_column,
        "record_count": expected_count,
    }


def run_discover(args: argparse.Namespace) -> int:
    if args.sample_size <= 0:
        raise ValueError("Sample size must be positive")
    if args.clusters is not None and args.clusters <= 0:
        raise ValueError("Cluster count must be positive")
    manifest_path = args.workspace / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise ValueError(
            f"Workspace already exists: {args.workspace}. Pass --force to archive it and create a fresh workspace."
        )
    if args.workspace.exists() and not manifest_path.exists() and any(args.workspace.iterdir()):
        raise ValueError(f"Refusing to replace non-workspace directory: {args.workspace}")
    records = _load_records(args.dataset)
    samples = select_diverse_samples(
        records,
        sample_size=args.sample_size,
        seed=args.seed,
        cluster_count=args.clusters,
        dimension_fields=args.dimension_fields,
    )
    archived_workspace = None
    if manifest_path.exists():
        archived_workspace = archive_review_workspace(args.workspace)
    initialize_review_workspace(
        args.workspace,
        records,
        samples,
        source=str(args.dataset),
        reviewer={
            key: value
            for key, value in {"name": args.reviewer, "role": args.reviewer_role}.items()
            if value
        },
        dimension_fields=args.dimension_fields,
    )
    coverage_count = sum(
        item["strategy"] in {"cluster_representative", "dimension_coverage"}
        for item in samples
    )
    print(f"Created review workspace: {args.workspace}")
    if archived_workspace:
        print(f"Archived previous workspace without modification: {archived_workspace}")
    print(f"Selected {len(samples)} of {len(records)} records ({coverage_count} coverage selections).")
    print("Discovery samples are coverage-biased; use a separate random sample for prevalence estimates.")
    print(f"Review samples: {args.workspace / 'samples.json'}")
    print(f"Record source: {args.workspace / 'records.jsonl'}")
    return 0


def run_promote(args: argparse.Namespace) -> int:
    workspace = load_review_workspace(args.workspace)
    taxonomy = workspace["taxonomy"]
    if args.failure_mode not in taxonomy:
        available = ", ".join(sorted(taxonomy)) or "none"
        raise ValueError(f"Unknown failure mode {args.failure_mode!r}. Available modes: {available}")
    fields = args.input_fields or ["input", "output", "reference", "trace"]
    failure_examples = []
    pass_examples = []
    for key, annotation in workspace["annotations"].items():
        critique = str(annotation.get("open_code") or annotation.get("critique") or "").strip()
        if not critique:
            continue
        try:
            record = workspace["records"][int(key)]
        except (ValueError, IndexError):
            continue
        example = {
            "annotation_id": str(key),
            "input": {field: record.get(field) for field in fields if field in record},
            "critique": critique,
            "result": (
                "Fail"
                if args.failure_mode in annotation.get("failure_modes", [])
                else "Pass"
            ),
        }
        if example["result"] == "Fail":
            failure_examples.append(example)
        elif annotation.get("pass_fail", True):
            pass_examples.append(example)
    expert_examples = [*failure_examples[:4], *pass_examples[:4]]
    source_ids = [example["annotation_id"] for example in expert_examples]
    evaluator = build_evaluator_definition(
        args.failure_mode,
        taxonomy[args.failure_mode],
        kind=args.kind,
        input_fields=fields,
        pass_definition=args.pass_definition,
        rule_pattern=args.pattern,
        source_annotation_ids=source_ids,
        expert_examples=expert_examples,
        errors_reviewed=args.errors_reviewed,
    )
    if args.out:
        _write_json(args.out, evaluator)
        target = args.out
    else:
        target = save_evaluator_definition(args.workspace, evaluator)
    print(f"Created draft evaluator: {target}")
    if not args.errors_reviewed:
        print("Checkpoint pending: review and fix obvious product errors before judge automation.")
    print("Status: draft_unvalidated. Validate it against held-out human labels before using it as a gate.")
    return 0


def run_split_labels(args: argparse.Namespace) -> int:
    targets = [args.out / f"{name}.jsonl" for name in ("train", "dev", "test")]
    if not args.force and any(path.exists() for path in targets):
        raise ValueError(f"Split files already exist in {args.out}. Pass --force to replace them.")
    records = _load_records(args.dataset)
    splits = split_labeled_records(
        records,
        args.label_column,
        seed=args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
    )
    artifacts = {}
    for name, records_for_split in splits.items():
        artifact_path = args.out / f"{name}.jsonl"
        _write_jsonl(artifact_path, records_for_split)
        provenance_fields = sorted(
            {field for record in records_for_split for field in record}
        )
        artifacts[name] = {
            "path": artifact_path.name,
            "record_count": len(records_for_split),
            "provenance_fields": provenance_fields,
            "records_sha256": _records_sha256(records_for_split, provenance_fields),
            "field_sha256": {
                field: _records_sha256(records_for_split, [field])
                for field in provenance_fields
            },
            "mutable_fields": [args.judge_column],
        }
    manifest = {
        "schema_version": "1.0",
        "source": str(args.dataset),
        "label_column": args.label_column,
        "judge_label_column": args.judge_column,
        "seed": args.seed,
        "counts": {name: len(items) for name, items in splits.items()},
        "artifacts": artifacts,
        "policy": {
            "train": "Few-shot examples and prompt construction only",
            "dev": "Evaluator iteration and threshold selection",
            "test": "One-way final validation and CI gating",
        },
    }
    _write_json(args.out / "split_manifest.json", manifest)
    print(f"Wrote disjoint stratified splits to: {args.out}")
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


def run_validate_judge(args: argparse.Namespace) -> int:
    if not 0 <= args.min_tpr <= 1 or not 0 <= args.min_tnr <= 1:
        raise ValueError("TPR and TNR thresholds must be between 0 and 1")
    if args.human_column == args.judge_column:
        raise ValueError("Human and judge label columns must be different")
    records = _load_records(args.dataset)
    provenance = _verify_test_split_provenance(
        args.dataset,
        records,
        args.split_manifest,
        args.human_column,
        args.judge_column,
    )
    missing = [
        column
        for column in (args.human_column, args.judge_column)
        if any(column not in record for record in records)
    ]
    if missing:
        raise ValueError(f"Missing required label columns: {', '.join(missing)}")
    human_labels = [record[args.human_column] for record in records]
    judge_labels = [record[args.judge_column] for record in records]
    metrics = validate_evaluator_labels(human_labels, judge_labels)
    intervals = bootstrap_metric_intervals(
        human_labels,
        judge_labels,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    gate_passed = (
        metrics["tpr"] is not None
        and metrics["tnr"] is not None
        and metrics["tpr"] >= args.min_tpr
        and metrics["tnr"] >= args.min_tnr
    )
    group_reports = []
    if args.group_by:
        grouped_records = {}
        for record in records:
            key = tuple(str(record.get(field, "(missing)")) for field in args.group_by)
            grouped_records.setdefault(key, []).append(record)
        for values, group in sorted(grouped_records.items()):
            group_reports.append(
                {
                    "dimensions": dict(zip(args.group_by, values)),
                    "metrics": validate_evaluator_labels(
                        [record[args.human_column] for record in group],
                        [record[args.judge_column] for record in group],
                    ),
                }
            )
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_role": "held_out_test",
        "provenance": provenance,
        "human_label_column": args.human_column,
        "evaluator_label_column": args.judge_column,
        "metrics": metrics,
        "confidence_intervals_95": intervals,
        "thresholds": {"minimum_tpr": args.min_tpr, "minimum_tnr": args.min_tnr},
        "gate_passed": gate_passed,
        "groups": group_reports,
    }
    evaluator = None
    if args.evaluator:
        evaluator = parse_evaluator_definition(
            json.loads(args.evaluator.read_text(encoding="utf-8"))
        )
        lifecycle = evaluator.lifecycle_checkpoints
        checkpoint_labels = {
            "human_review_completed": "human review is incomplete",
            "obvious_errors_reviewed": "obvious product errors were not reviewed",
            "expert_examples_include_pass_and_fail": (
                "expert examples do not include both Pass and Fail critiques"
            ),
        }
        incomplete_checkpoints = [
            message
            for checkpoint, message in checkpoint_labels.items()
            if getattr(lifecycle, checkpoint) is not True
        ]
        if incomplete_checkpoints:
            gate_passed = False
            report["gate_passed"] = False
            report["gate_blockers"] = incomplete_checkpoints

    typed_report = JudgeValidationReport.model_validate(report)
    if evaluator is not None:
        evaluator.status = "validated" if gate_passed else "validation_failed"
        evaluator.validation.validated = gate_passed
        evaluator.validation.last_report = typed_report
        evaluator.validation.history.append(typed_report)
        _write_json(args.evaluator, dump_evaluator_definition(evaluator))
        print(f"Updated evaluator status: {args.evaluator}")
    if args.out:
        _write_json(args.out, typed_report.model_dump(mode="json", by_alias=True))
        print(f"Wrote validation report: {args.out}")
    print(
        "Judge validation: "
        f"TPR={metrics['tpr'] if metrics['tpr'] is not None else 'n/a'}, "
        f"TNR={metrics['tnr'] if metrics['tnr'] is not None else 'n/a'}, "
        f"balanced_accuracy={metrics['balanced_accuracy'] if metrics['balanced_accuracy'] is not None else 'n/a'}"
    )
    print("Gate: PASS" if gate_passed else "Gate: FAIL")
    return 0 if gate_passed else 1


def _available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def doctor(_args: argparse.Namespace) -> int:
    root = app_dir()
    checks = {
        "streamlit": _available("streamlit"),
        "pandas": _available("pandas"),
        "openai": _available("openai"),
        "anthropic": _available("anthropic"),
        "sentence_transformers": _available("sentence_transformers"),
        "mteb": _available("mteb"),
        "torch": _available("torch"),
        "z3": _available("z3"),
    }

    print(f"Vectory CLI {__version__}")
    print(f"Python {sys.version.split()[0]}")
    print(f"App files: {root}")
    print()
    for name, ok in checks.items():
        marker = "ok" if ok else "missing"
        print(f"{name}: {marker}")
    print()
    print("Embedding comparison features are optional.")
    print("For source checkouts: pip install -e '.[embedding]'")
    print("For pipx installs: add torch, mteb, and sentence-transformers to the Vectory environment.")
    print()
    print("Formal checker execution is optional and disabled by default.")
    print("Install formal helpers with: pip install 'vectoryai[formal]'")
    print("Run trusted local checkers with: vectory benchmark SUBMISSION --workspace PATH --allow-formal-runtime")
    return 0 if checks["streamlit"] and checks["pandas"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vectory",
        description="Discover AI failures, validate evaluators, and score Vectory Benchmark submissions.",
    )
    parser.add_argument("--version", action="version", version=f"Vectory CLI {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    app_parser = subparsers.add_parser("app", help="Launch the local Vectory Streamlit app.")
    app_parser.add_argument("--port", type=int, default=8501, help="Streamlit port. Default: 8501.")
    app_parser.add_argument("--address", default="localhost", help="Bind address. Default: localhost.")
    app_parser.add_argument("--headless", action="store_true", help="Run Streamlit without opening a browser.")
    app_parser.add_argument("streamlit_args", nargs=argparse.REMAINDER, help="Extra arguments passed to Streamlit.")
    app_parser.set_defaults(func=run_app)

    benchmark_parser = subparsers.add_parser("benchmark", help="Score a Vectory Benchmark submission.")
    benchmark_parser.add_argument("submission", type=Path, help="Path to a JSON or JSONL benchmark submission.")
    benchmark_parser.add_argument("--suite", type=Path, help="Optional benchmark suite manifest path.")
    benchmark_parser.add_argument("--scores-out", type=Path, help="Optional run-score output path, .json or .csv.")
    benchmark_parser.add_argument("--leaderboard-out", type=Path, help="Optional leaderboard output path, .json or .csv.")
    benchmark_parser.add_argument("--report-out", type=Path, help="Optional static report bundle output directory.")
    benchmark_parser.add_argument("--gate-min-score", type=float, help="Fail if any run is below this score.")
    benchmark_parser.add_argument(
        "--gate-block-severity",
        choices=["low", "medium", "high", "critical"],
        default="critical",
        help="Fail when a pathology at or above this severity appears. Default: critical.",
    )
    benchmark_parser.add_argument("--gate-max-pathology-risk", type=float, help="Fail when aggregate pathology risk exceeds this value.")
    benchmark_parser.add_argument("--workspace", type=Path, help="Workspace for trusted suite-defined formal checker commands.")
    benchmark_parser.add_argument("--allow-formal-runtime", action="store_true", help="Run trusted formal checker commands from the local suite manifest.")
    benchmark_parser.set_defaults(func=run_benchmark)

    gate_parser = subparsers.add_parser("gate", help="Run a CI-style Vectory Benchmark release gate.")
    gate_parser.add_argument("submission", type=Path, help="Path to a JSON or JSONL benchmark submission.")
    gate_parser.add_argument("--suite", type=Path, help="Optional benchmark suite manifest path.")
    gate_parser.add_argument("--min-score", type=float, default=0.86, help="Minimum per-run Vectory score. Default: 0.86.")
    gate_parser.add_argument(
        "--block-severity",
        choices=["low", "medium", "high", "critical"],
        default="critical",
        help="Block pathologies at or above this severity. Default: critical.",
    )
    gate_parser.add_argument("--max-pathology-risk", type=float, help="Maximum aggregate pathology risk per run.")
    gate_parser.add_argument("--report-out", type=Path, help="Optional static report bundle output directory.")
    gate_parser.set_defaults(func=run_gate)

    discover_parser = subparsers.add_parser(
        "discover",
        help="Create a diverse, human-reviewable workspace from AI outputs or traces.",
    )
    discover_parser.add_argument("dataset", type=Path, help="JSON, JSONL, or CSV outputs/traces.")
    discover_parser.add_argument("--workspace", type=Path, required=True, help="Review workspace directory.")
    discover_parser.add_argument("--sample-size", type=int, default=30, help="Records selected for discovery. Default: 30.")
    discover_parser.add_argument("--clusters", type=int, help="Optional cluster count; inferred when omitted.")
    discover_parser.add_argument(
        "--dimension-fields",
        nargs="+",
        help="Optional product dimensions to cover, such as feature scenario persona.",
    )
    discover_parser.add_argument("--reviewer", help="Principal domain expert name or stable identifier.")
    discover_parser.add_argument("--reviewer-role", help="Principal domain expert role.")
    discover_parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed. Default: 42.")
    discover_parser.add_argument(
        "--force",
        action="store_true",
        help="Archive the complete existing workspace, then create a fresh workspace.",
    )
    discover_parser.set_defaults(func=run_discover)

    promote_parser = subparsers.add_parser(
        "promote",
        help="Promote an accepted failure mode into a draft binary evaluator.",
    )
    promote_parser.add_argument("workspace", type=Path, help="Vectory review workspace.")
    promote_parser.add_argument("failure_mode", help="Exact accepted taxonomy name.")
    promote_parser.add_argument("--kind", choices=["llm_judge", "regex", "contains"], default="llm_judge")
    promote_parser.add_argument("--input-fields", nargs="+", help="Fields supplied to the evaluator.")
    promote_parser.add_argument("--pass-definition", help="Explicit definition of a passing result.")
    promote_parser.add_argument("--pattern", help="Required pattern for regex or contains evaluators.")
    promote_parser.add_argument(
        "--errors-reviewed",
        action="store_true",
        help="Confirm obvious product errors were fixed or reviewed before judge automation.",
    )
    promote_parser.add_argument("--out", type=Path, help="Optional output path outside the workspace.")
    promote_parser.set_defaults(func=run_promote)

    split_parser = subparsers.add_parser(
        "split-labels",
        help="Create disjoint stratified train, development, and test label sets.",
    )
    split_parser.add_argument("dataset", type=Path, help="JSON, JSONL, or CSV with trusted human labels.")
    split_parser.add_argument("--label-column", required=True, help="Binary Pass/Fail human-label column.")
    split_parser.add_argument(
        "--judge-column",
        required=True,
        help="Prediction column that may be added or replaced after splitting.",
    )
    split_parser.add_argument("--out", type=Path, required=True, help="Output directory for split JSONL files.")
    split_parser.add_argument("--train-fraction", type=float, default=0.15)
    split_parser.add_argument("--dev-fraction", type=float, default=0.45)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--force", action="store_true", help="Replace existing split files.")
    split_parser.set_defaults(func=run_split_labels)

    validate_parser = subparsers.add_parser(
        "validate-judge",
        help="Validate binary evaluator labels against a held-out human-labeled test set.",
    )
    validate_parser.add_argument("dataset", type=Path, help="Held-out JSON, JSONL, or CSV test set.")
    validate_parser.add_argument("--human-column", required=True, help="Trusted human Pass/Fail label column.")
    validate_parser.add_argument("--judge-column", required=True, help="Evaluator Pass/Fail label column.")
    validate_parser.add_argument("--min-tpr", type=float, default=0.8, help="Minimum true-pass rate. Default: 0.8.")
    validate_parser.add_argument("--min-tnr", type=float, default=0.8, help="Minimum true-fail rate. Default: 0.8.")
    validate_parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    validate_parser.add_argument("--seed", type=int, default=42)
    validate_parser.add_argument("--out", type=Path, help="Optional JSON validation report path.")
    validate_parser.add_argument("--evaluator", type=Path, help="Optional evaluator JSON whose status should be updated.")
    validate_parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Manifest used to verify that the dataset is the untouched held-out test partition.",
    )
    validate_parser.add_argument(
        "--group-by",
        nargs="+",
        help="Report metrics by dimensions such as feature, scenario, or persona.",
    )
    validate_parser.set_defaults(func=run_validate_judge)

    doctor_parser = subparsers.add_parser("doctor", help="Check local Vectory installation health.")
    doctor_parser.set_defaults(func=doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
