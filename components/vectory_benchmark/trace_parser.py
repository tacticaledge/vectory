"""Trace parsing and normalization helpers for VectoryBenchmark."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from components.vectory_benchmark.schemas import AgentRun, EventType, TraceEvent


def load_submission_payload(source: str | Path | bytes) -> list[dict[str, Any]]:
    """Load a JSON or JSONL submission payload."""
    if isinstance(source, bytes):
        text = source.decode("utf-8")
    else:
        candidate = Path(source)
        text = candidate.read_text(encoding="utf-8") if candidate.exists() else str(source)

    stripped = text.strip()
    if not stripped:
        return []

    if stripped.startswith("[") or stripped.startswith("{"):
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            if "runs" in payload:
                payload = payload["runs"]
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("Submission JSON must be an object, a list, or an object with a runs list")
        return payload

    rows = []
    for line_number, line in enumerate(stripped.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
    return rows


def normalize_event(raw: dict[str, Any] | TraceEvent) -> TraceEvent:
    """Normalize common trajectory shapes into a TraceEvent."""
    if isinstance(raw, TraceEvent):
        return raw

    event = dict(raw)
    raw_type = str(event.get("type") or event.get("event_type") or "").lower()
    name = event.get("name") or event.get("tool_name") or event.get("command")
    content = event.get("content") or event.get("message") or event.get("text") or ""

    if not raw_type:
        if name or "input" in event or "arguments" in event:
            raw_type = EventType.TOOL_CALL.value
        elif "path" in event or "diff" in event:
            raw_type = EventType.FILE_EDIT.value
        else:
            raw_type = EventType.MESSAGE.value

    aliases = {
        "assistant": EventType.MESSAGE,
        "user": EventType.MESSAGE,
        "thought": EventType.MESSAGE,
        "tool": EventType.TOOL_CALL,
        "tool_call": EventType.TOOL_CALL,
        "tool_result": EventType.TOOL_RESULT,
        "observation": EventType.TOOL_RESULT,
        "edit": EventType.FILE_EDIT,
        "file_edit": EventType.FILE_EDIT,
        "test": EventType.TEST_RUN,
        "test_run": EventType.TEST_RUN,
        "verify": EventType.VERIFICATION,
        "verification": EventType.VERIFICATION,
        "decision": EventType.DECISION,
        "claim": EventType.CLAIM,
        "evidence": EventType.EVIDENCE,
        "proof": EventType.PROOF_OBLIGATION,
        "proof_obligation": EventType.PROOF_OBLIGATION,
        "obligation": EventType.PROOF_OBLIGATION,
        "checker": EventType.CHECKER_RESULT,
        "checker_result": EventType.CHECKER_RESULT,
        "policy": EventType.POLICY_CHECK,
        "policy_check": EventType.POLICY_CHECK,
        "checkpoint": EventType.CHECKPOINT,
        "snapshot": EventType.CHECKPOINT,
        "progress": EventType.CHECKPOINT,
        "error": EventType.ERROR,
        "final": EventType.FINAL,
    }

    event_type = aliases.get(raw_type, EventType.MESSAGE)
    normalized = {
        "type": event_type,
        "content": str(content),
        "name": name,
        "input": event.get("input", event.get("arguments")),
        "output": event.get("output", event.get("result")),
        "path": event.get("path") or event.get("file"),
        "success": event.get("success"),
        "timestamp": event.get("timestamp"),
        "metadata": event.get("metadata") or {},
    }

    return TraceEvent(**normalized)


def normalize_run(raw: dict[str, Any] | AgentRun) -> AgentRun:
    """Normalize a submitted run into the canonical schema."""
    if isinstance(raw, AgentRun):
        return raw

    run = dict(raw)
    events = [normalize_event(event) for event in run.get("events", [])]
    final_answer = run.get("final_answer") or run.get("answer") or run.get("final_result") or ""
    agent = run.get("agent") or run.get("agent_name") or "unknown-agent"
    model = run.get("model") or run.get("model_name") or "unknown-model"
    task_id = run.get("task_id") or run.get("query") or run.get("case_id") or ""
    run_id = run.get("run_id") or run.get("id") or f"{task_id}.run"

    return AgentRun(
        agent=str(agent),
        model=str(model),
        task_id=str(task_id),
        run_id=str(run_id),
        final_answer=str(final_answer),
        status=run.get("status", "unknown"),
        events=events,
        claims=run.get("claims") or [],
        evidence=run.get("evidence") or run.get("evidence_references") or [],
        proof_obligations=run.get("proof_obligations") or run.get("obligations") or [],
        checker_results=run.get("checker_results") or run.get("checks") or [],
        metrics=run.get("metrics") or {},
        declared_success=run.get("declared_success"),
    )


def normalize_runs(raw_runs: Iterable[dict[str, Any] | AgentRun]) -> list[AgentRun]:
    """Normalize many submitted runs."""
    return [normalize_run(raw_run) for raw_run in raw_runs]
