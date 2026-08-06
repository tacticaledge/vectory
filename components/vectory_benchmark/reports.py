"""Report bundle and release gate helpers for Vectory Benchmark."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from components.vectory_benchmark.schemas import (
    AgentRun,
    BenchmarkSuite,
    BenchmarkTask,
    EventType,
    RunScore,
    Severity,
)
from components.vectory_benchmark.trace_parser import normalize_run


SEVERITY_ORDER = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


@dataclass(frozen=True)
class GateResult:
    """Outcome of a release gate evaluation."""

    passed: bool
    reasons: list[str]
    min_score: float
    block_severity: Severity
    max_pathology_risk: float | None = None


def score_rows(scores: list[RunScore]) -> list[dict[str, Any]]:
    """Flatten run scores for JSON/CSV export."""

    rows: list[dict[str, Any]] = []
    for score in scores:
        row: dict[str, Any] = {
            "agent": score.agent,
            "model": score.model,
            "task_id": score.task_id,
            "domain": score.domain.value,
            "run_id": score.run_id,
            "vectory_score": score.vectory_score,
            "band": score.band.value,
            "passed": score.passed,
            "pathology_count": len(score.pathologies),
            "pathology_codes": ",".join(finding.code for finding in score.pathologies),
        }
        for name, dimension in score.dimensions.items():
            row[name] = round(dimension.score, 4)
        rows.append(row)
    return rows


def pathology_rows(scores: list[RunScore]) -> list[dict[str, Any]]:
    """Flatten pathology findings for report artifacts."""

    rows: list[dict[str, Any]] = []
    for score in scores:
        for finding in score.pathologies:
            rows.append(
                {
                    "agent": score.agent,
                    "model": score.model,
                    "task_id": score.task_id,
                    "run_id": score.run_id,
                    "code": finding.code,
                    "name": finding.name,
                    "severity": finding.severity.value,
                    "penalty": finding.score_penalty,
                    "evidence": "; ".join(finding.evidence),
                    "recommendation": finding.recommendation,
                }
            )
    return rows


def claim_evidence_rows(runs: list[AgentRun | dict[str, Any]], scores: list[RunScore]) -> list[dict[str, Any]]:
    """Build claim-to-evidence rows from submitted traces."""

    score_by_run_id = {score.run_id: score for score in scores}
    rows: list[dict[str, Any]] = []
    for raw_run in runs:
        run = normalize_run(raw_run)
        score = score_by_run_id.get(run.run_id)
        evidence_by_id = {evidence.evidence_id: evidence for evidence in run.evidence}
        obligation_by_id = {obligation.obligation_id: obligation for obligation in run.proof_obligations}
        checker_by_obligation: dict[str, list[str]] = {}
        for result in run.checker_results:
            for obligation_id in result.obligation_ids:
                checker_by_obligation.setdefault(obligation_id, []).append(
                    f"{result.checker_id}:{result.status}"
                )
        for claim in run.claims:
            evidence_status = []
            for evidence_id in claim.evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    evidence_status.append(f"{evidence_id}:missing")
                else:
                    evidence_status.append(f"{evidence_id}:{evidence.source_type}")
            obligation_status = []
            for obligation_id in claim.obligation_ids:
                obligation = obligation_by_id.get(obligation_id)
                if obligation is None:
                    obligation_status.append(f"{obligation_id}:missing")
                else:
                    checkers = ",".join(checker_by_obligation.get(obligation_id, []))
                    suffix = f";checkers={checkers}" if checkers else ""
                    obligation_status.append(f"{obligation_id}:{obligation.status}{suffix}")
            rows.append(
                {
                    "agent": run.agent,
                    "model": run.model,
                    "task_id": run.task_id,
                    "run_id": run.run_id,
                    "claim_id": claim.claim_id,
                    "claim": claim.text,
                    "evidence": "; ".join(evidence_status),
                    "obligations": "; ".join(obligation_status),
                    "score": score.vectory_score if score else None,
                    "passed": score.passed if score else None,
                }
            )
    return rows


def checkpoint_rows(runs: list[AgentRun | dict[str, Any]], scores: list[RunScore]) -> list[dict[str, Any]]:
    """Extract submitted checkpoint events for timeline reporting."""

    score_by_run_id = {score.run_id: score for score in scores}
    rows: list[dict[str, Any]] = []
    for raw_run in runs:
        run = normalize_run(raw_run)
        final_score = score_by_run_id.get(run.run_id)
        for index, event in enumerate(run.events):
            if event.type != EventType.CHECKPOINT:
                continue
            payload = event.output if isinstance(event.output, dict) else {}
            metadata = event.metadata if isinstance(event.metadata, dict) else {}
            wall_time = payload.get("wall_time_seconds", metadata.get("wall_time_seconds"))
            score_value = payload.get("score", metadata.get("score"))
            open_obligations = payload.get("open_obligations", metadata.get("open_obligations"))
            failed_checkers = payload.get("failed_checkers", metadata.get("failed_checkers"))
            rows.append(
                {
                    "agent": run.agent,
                    "model": run.model,
                    "task_id": run.task_id,
                    "run_id": run.run_id,
                    "checkpoint_index": index,
                    "name": event.name or "",
                    "wall_time_seconds": wall_time,
                    "checkpoint_score": score_value,
                    "open_obligations": open_obligations,
                    "failed_checkers": failed_checkers,
                    "final_score": final_score.vectory_score if final_score else None,
                    "content": event.content,
                }
            )
    return rows


def benchmark_card(suite: BenchmarkSuite, tasks: list[BenchmarkTask]) -> dict[str, Any]:
    """Create a benchmark-card artifact for the scored suite/tasks."""

    return {
        "suite_id": suite.suite_id,
        "version": suite.version,
        "title": suite.title,
        "description": suite.description,
        "task_count": len(tasks),
        "tasks": [
            {
                "task_id": task.task_id,
                "title": task.title,
                "domain": task.domain.value,
                "difficulty": task.difficulty,
                "intent": task.intent,
                "success_criteria": task.success_criteria,
                "required_event_types": [event.value for event in task.checks.required_event_types],
                "required_tools": task.checks.required_tools,
                "required_artifacts": task.checks.required_artifacts,
                "requires_proof": task.checks.requires_proof,
                "min_proof_coverage": task.checks.min_proof_coverage,
                "accepted_checker_types": task.checks.accepted_checker_types,
                "limits": task.limits.model_dump(),
                "weights": task.weights.model_dump(),
                "tags": task.tags,
            }
            for task in tasks
        ],
        "trust_boundary": {
            "submitted_traces_may_provide_checker_output": True,
            "submitted_traces_may_define_executable_commands": False,
            "trusted_checker_commands_source": "suite manifest only",
            "formal_runtime_default": "disabled",
        },
    }


def evaluate_gate(
    scores: list[RunScore],
    *,
    min_score: float = 0.86,
    block_severity: Severity = Severity.CRITICAL,
    max_pathology_risk: float | None = None,
) -> GateResult:
    """Evaluate a CI/release gate over scored runs."""

    reasons: list[str] = []
    threshold = SEVERITY_ORDER[block_severity]
    for score in scores:
        if score.vectory_score < min_score:
            reasons.append(
                f"{score.run_id} score {score.vectory_score:.3f} below min {min_score:.3f}"
            )
        if not score.passed:
            reasons.append(f"{score.run_id} did not pass task gate")
        risk = min(1.0, sum(finding.score_penalty for finding in score.pathologies))
        if max_pathology_risk is not None and risk > max_pathology_risk:
            reasons.append(
                f"{score.run_id} pathology risk {risk:.3f} above max {max_pathology_risk:.3f}"
            )
        for finding in score.pathologies:
            if SEVERITY_ORDER[finding.severity] >= threshold:
                reasons.append(
                    f"{score.run_id} has {finding.severity.value} pathology {finding.code}"
                )
    return GateResult(
        passed=not reasons,
        reasons=reasons,
        min_score=min_score,
        block_severity=block_severity,
        max_pathology_risk=max_pathology_risk,
    )


def write_report_bundle(
    output_dir: Path,
    *,
    suite: BenchmarkSuite,
    tasks: list[BenchmarkTask],
    runs: list[AgentRun | dict[str, Any]],
    scores: list[RunScore],
    leaderboard_rows: list[dict[str, Any]],
    gate: GateResult | None = None,
) -> None:
    """Write a static benchmark report bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    score_data = score_rows(scores)
    pathology_data = pathology_rows(scores)
    claim_data = claim_evidence_rows(runs, scores)
    checkpoint_data = checkpoint_rows(runs, scores)
    card = benchmark_card(suite, tasks)

    _write_json(output_dir / "scores.json", score_data)
    _write_json(output_dir / "leaderboard.json", leaderboard_rows)
    _write_json(output_dir / "pathologies.json", pathology_data)
    _write_json(output_dir / "claim_evidence_table.json", claim_data)
    _write_json(output_dir / "checkpoints.json", checkpoint_data)
    _write_json(output_dir / "benchmark_card.json", card)
    if gate is not None:
        _write_json(output_dir / "gate.json", gate.__dict__)

    _write_csv(output_dir / "scores.csv", score_data)
    _write_csv(output_dir / "pathologies.csv", pathology_data)
    _write_csv(output_dir / "claim_evidence_table.csv", claim_data)
    _write_csv(output_dir / "checkpoints.csv", checkpoint_data)
    (output_dir / "index.html").write_text(
        _render_html_report(
            suite=suite,
            scores=scores,
            score_data=score_data,
            pathology_data=pathology_data,
            claim_data=claim_data,
            checkpoint_data=checkpoint_data,
            gate=gate,
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(
            {_sanitize_csv_cell(key): _sanitize_csv_cell(value) for key, value in row.items()}
            for row in rows
        )


def _sanitize_csv_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("'"):
        return value
    stripped = value.lstrip()
    if value[:1] in {"=", "+", "-", "@", "\t", "\r", "\n"} or stripped[:1] in {"=", "+", "-", "@"}:
        return f"'{value}"
    return value


def _render_html_report(
    *,
    suite: BenchmarkSuite,
    scores: list[RunScore],
    score_data: list[dict[str, Any]],
    pathology_data: list[dict[str, Any]],
    claim_data: list[dict[str, Any]],
    checkpoint_data: list[dict[str, Any]],
    gate: GateResult | None,
) -> str:
    mean_score = sum(score.vectory_score for score in scores) / len(scores) if scores else 0.0
    passed = sum(1 for score in scores if score.passed)
    gate_label = "not evaluated"
    if gate is not None:
        gate_label = "passed" if gate.passed else "failed"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(suite.title)} | Vectory Benchmark Report</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: #0b0d14; color: #f4f6fb; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 40px 20px 64px; }}
    h1 {{ font-size: 36px; margin: 0 0 10px; }}
    h2 {{ margin-top: 36px; }}
    .muted {{ color: #a4adbd; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .card {{ border: 1px solid #2b3140; border-radius: 8px; padding: 16px; background: #121622; }}
    .value {{ font-size: 28px; font-weight: 750; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #2b3140; padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #a4adbd; font-weight: 650; }}
    code {{ color: #95a4ff; }}
    a {{ color: #8ea2ff; }}
  </style>
</head>
<body>
<main>
  <p class="muted">Vectory Benchmark Report</p>
  <h1>{html.escape(suite.title)}</h1>
  <p class="muted">{html.escape(suite.description)}</p>
  <div class="grid">
    <div class="card"><div class="muted">Runs scored</div><div class="value">{len(scores)}</div></div>
    <div class="card"><div class="muted">Pass@1</div><div class="value">{(passed / len(scores)) if scores else 0:.1%}</div></div>
    <div class="card"><div class="muted">Mean score</div><div class="value">{mean_score:.3f}</div></div>
    <div class="card"><div class="muted">Gate</div><div class="value">{html.escape(gate_label)}</div></div>
  </div>
  <h2>Run Scores</h2>
  {_html_table(score_data[:50])}
  <h2>Pathologies</h2>
  {_html_table(pathology_data[:50]) if pathology_data else '<p class="muted">None detected.</p>'}
  <h2>Claim Evidence</h2>
  {_html_table(claim_data[:50]) if claim_data else '<p class="muted">No claims submitted.</p>'}
  <h2>Checkpoints</h2>
  {_html_table(checkpoint_data[:50]) if checkpoint_data else '<p class="muted">No checkpoint events submitted.</p>'}
  <h2>Artifacts</h2>
  <p class="muted">This bundle includes <code>scores.json</code>, <code>leaderboard.json</code>, <code>pathologies.json</code>, <code>claim_evidence_table.json</code>, <code>checkpoints.json</code>, and <code>benchmark_card.json</code>.</p>
</main>
</body>
</html>
"""


def _html_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">No rows.</p>'
    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(column, '')))}</td>"
                for column in columns
            )
            + "</tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
