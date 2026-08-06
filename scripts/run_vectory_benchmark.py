#!/usr/bin/env python3
"""Run VectoryBenchmark scoring from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from components.vectory_benchmark import build_leaderboard, load_suite, score_submission
    from components.vectory_benchmark.formal_runtime import run_trusted_formal_checkers
    from components.vectory_benchmark.reports import (
        evaluate_gate,
        score_rows,
        write_report_bundle,
    )
    from components.vectory_benchmark.schemas import Severity
    from components.vectory_benchmark.trace_parser import load_submission_payload, normalize_run
except ModuleNotFoundError as exc:
    missing = exc.name or "a required dependency"
    print(
        f"Missing dependency: {missing}. Install Vectory with `pip install vectoryai` "
        "or install this checkout with `pip install -e .`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(scores, leaderboard) -> None:
    total = len(scores)
    passed = sum(1 for score in scores if score.passed)
    mean_score = sum(score.vectory_score for score in scores) / total if total else 0.0
    pathology_counts = Counter(
        finding.code
        for score in scores
        for finding in score.pathologies
    )

    print("VectoryBenchmark")
    print(f"Runs scored: {total}")
    print(f"Pass@1: {passed / total:.1%}" if total else "Pass@1: n/a")
    print(f"Mean Vectory Score: {mean_score:.3f}")

    if not leaderboard.empty:
        top = leaderboard.iloc[0]
        print(f"Top agent: {top['agent']} / {top['model']} ({top['vectory_score']:.3f})")

    if pathology_counts:
        print("Pathologies:")
        for code, count in pathology_counts.most_common():
            print(f"  {code}: {count}")
    else:
        print("Pathologies: none detected")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score submitted agent traces with VectoryBenchmark."
    )
    parser.add_argument(
        "submission",
        type=Path,
        help="Path to a JSON or JSONL submission file.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="Optional path to a VectoryBenchmark suite manifest.",
    )
    parser.add_argument(
        "--scores-out",
        type=Path,
        default=None,
        help="Optional path for run scores. Supports .json or .csv.",
    )
    parser.add_argument(
        "--leaderboard-out",
        type=Path,
        default=None,
        help="Optional path for leaderboard output. Supports .json or .csv.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional directory for a static report bundle.",
    )
    parser.add_argument(
        "--gate-min-score",
        type=float,
        default=None,
        help="Fail with exit code 1 if any run is below this score.",
    )
    parser.add_argument(
        "--gate-block-severity",
        choices=[severity.value for severity in Severity],
        default="critical",
        help="Fail when a pathology at or above this severity appears. Default: critical.",
    )
    parser.add_argument(
        "--gate-max-pathology-risk",
        type=float,
        default=None,
        help="Fail when a run's aggregate pathology penalty exceeds this value.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace used for trusted suite-defined formal checker commands.",
    )
    parser.add_argument(
        "--allow-formal-runtime",
        action="store_true",
        help="Run trusted formal checker commands from the local suite manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    suite = load_suite(args.suite)
    payload = load_submission_payload(args.submission)
    if args.allow_formal_runtime:
        if args.workspace is None:
            raise SystemExit("--allow-formal-runtime requires --workspace")
        task_by_id = {task.task_id: task for task in suite.tasks}
        enriched_payload = []
        for raw_run in payload:
            run = normalize_run(raw_run)
            if run.task_id not in task_by_id:
                raise ValueError(f"Unknown task_id in submission: {run.task_id}")
            enriched_payload.append(run_trusted_formal_checkers(task_by_id[run.task_id], run, args.workspace))
        payload = enriched_payload
    scores = score_submission(suite.tasks, payload)
    leaderboard = build_leaderboard(scores)
    rows = score_rows(scores)
    gate = None
    if args.gate_min_score is not None or args.gate_max_pathology_risk is not None:
        gate = evaluate_gate(
            scores,
            min_score=args.gate_min_score if args.gate_min_score is not None else 0.0,
            block_severity=Severity(args.gate_block_severity),
            max_pathology_risk=args.gate_max_pathology_risk,
        )

    _print_summary(scores, leaderboard)

    if args.scores_out:
        if args.scores_out.suffix.lower() == ".csv":
            import pandas as pd

            args.scores_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.scores_out, index=False)
        else:
            _write_json(args.scores_out, rows)
        print(f"Wrote run scores: {args.scores_out}")

    if args.leaderboard_out:
        if args.leaderboard_out.suffix.lower() == ".csv":
            args.leaderboard_out.parent.mkdir(parents=True, exist_ok=True)
            leaderboard.to_csv(args.leaderboard_out, index=False)
        else:
            _write_json(args.leaderboard_out, leaderboard.to_dict(orient="records"))
        print(f"Wrote leaderboard: {args.leaderboard_out}")

    if args.report_out:
        write_report_bundle(
            args.report_out,
            suite=suite,
            tasks=suite.tasks,
            runs=payload,
            scores=scores,
            leaderboard_rows=leaderboard.to_dict(orient="records"),
            gate=gate,
        )
        print(f"Wrote report bundle: {args.report_out}")

    if gate is not None:
        if gate.passed:
            print("Gate: passed")
        else:
            print("Gate: failed")
            for reason in gate.reasons:
                print(f"  {reason}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
