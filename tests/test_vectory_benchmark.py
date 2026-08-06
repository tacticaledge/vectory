"""
Tests for VectoryBenchmark scoring, pathologies, and aggregation.
"""

import csv
from pathlib import Path

import pytest

from components.vectory_benchmark.leaderboard import build_leaderboard
from components.vectory_benchmark.pathology import detect_pathologies
from components.vectory_benchmark.reports import (
    evaluate_gate,
    write_report_bundle,
)
from components.vectory_benchmark.schemas import AgentRun, BenchmarkTask, EventType, Severity, TaskChecks, TaskLimits
from components.vectory_benchmark.scoring import score_run, score_submission
from components.vectory_benchmark.suite import load_suite
from components.vectory_benchmark.trace_parser import load_submission_payload, normalize_run


def test_load_default_suite_has_unique_tasks():
    suite = load_suite()

    assert suite.suite_id == "vectorybenchmark-core"
    assert len(suite.tasks) >= 8
    assert len({task.task_id for task in suite.tasks}) == len(suite.tasks)


def test_score_run_rewards_grounded_completed_trace():
    task = BenchmarkTask(
        task_id="coding.test",
        title="Coding Test",
        domain="coding",
        intent="Fix code after inspection.",
        setup="Inspect, edit, verify.",
        success_criteria=["fixed"],
        checks=TaskChecks(
            expected_answer_keywords=["fix", "test"],
            required_event_types=[EventType.TOOL_CALL, EventType.FILE_EDIT, EventType.VERIFICATION],
            reality_sampling_tools=["read_file"],
            required_artifacts=["src/parser.py"],
            min_reality_events=1,
            requires_verification=True,
        ),
        limits=TaskLimits(max_events=20, max_tool_calls=10, turn_cliff_events=12),
    )
    run = {
        "agent": "GroundedAgent",
        "model": "local",
        "task_id": "coding.test",
        "run_id": "run_0",
        "status": "completed",
        "final_answer": "Implemented the fix in src/parser.py and verified the test.",
        "declared_success": True,
        "events": [
            {"type": "tool_call", "name": "read_file", "path": "tests/test_parser.py", "success": True},
            {"type": "file_edit", "path": "src/parser.py", "success": True},
            {"type": "verification", "name": "pytest", "success": True},
            {"type": "final", "content": "done"},
        ],
    }

    score = score_run(task, run)

    assert score.passed is True
    assert score.vectory_score >= 0.85
    assert score.dimensions["reality_sampling"].score == 1.0
    assert score.pathologies == []


def test_detects_premature_planning_and_missing_sampling():
    task = BenchmarkTask(
        task_id="data.test",
        title="Data Test",
        domain="data_reasoning",
        intent="Answer after sampling.",
        setup="Must sample first.",
        success_criteria=["sample"],
        checks=TaskChecks(min_reality_events=2, penalize_premature_planning=True),
        limits=TaskLimits(max_events=20, max_tool_calls=10, turn_cliff_events=12),
    )
    run = AgentRun(
        agent="Planner",
        model="local",
        task_id="data.test",
        run_id="run_0",
        status="completed",
        final_answer="I computed the answer.",
        events=[
            {"type": "message", "content": "I will plan the approach first, then inspect data later."},
            {"type": "final", "content": "done"},
        ],
    )

    codes = {finding.code for finding in detect_pathologies(task, run)}

    assert "premature_planning" in codes
    assert "insufficient_reality_sampling" in codes


def test_detects_retrieval_thrashing_and_turn_cliff():
    task = BenchmarkTask(
        task_id="research.test",
        title="Research Test",
        domain="research",
        intent="Search with convergence.",
        setup="Must retrieve evidence.",
        success_criteria=["source"],
        checks=TaskChecks(retrieval_tools=["search"], requires_retrieval=True, min_reality_events=1),
        limits=TaskLimits(max_events=20, max_tool_calls=12, turn_cliff_events=8),
    )
    events = [
        {"type": "tool_call", "name": "search", "input": {"query": "same query"}, "success": True}
        for _ in range(9)
    ]
    run = AgentRun(
        agent="Searcher",
        model="local",
        task_id="research.test",
        run_id="run_0",
        status="completed",
        final_answer="source",
        events=events,
    )

    score = score_run(task, run)
    codes = {finding.code for finding in score.pathologies}

    assert "retrieval_thrashing" in codes
    assert "search_without_convergence" in codes
    assert "turn_cliff_decay" in codes
    assert score.dimensions["tool_retrieval_discipline"].score < 0.8


def test_score_submission_and_leaderboard():
    suite = load_suite()
    task = suite.tasks[0]
    run = {
        "agent": "ExampleAgent",
        "model": "example-model",
        "task_id": task.task_id,
        "run_id": "run_0",
        "status": "completed",
        "final_answer": "test fix src/parser.py",
        "events": [
            {"type": "tool_call", "name": "read_file", "path": "tests/test_parser.py", "success": True},
            {"type": "tool_call", "name": "rg", "success": True},
            {"type": "file_edit", "path": "src/parser.py", "success": True},
            {"type": "verification", "name": "pytest", "success": True},
        ],
    }

    scores = score_submission(suite.tasks, [run])
    leaderboard = build_leaderboard(scores)

    assert len(scores) == 1
    assert leaderboard.iloc[0]["agent"] == "ExampleAgent"
    assert leaderboard.iloc[0]["runs"] == 1
    assert "agent_control_index" in leaderboard.columns


def test_trace_parser_accepts_jsonl_and_aliases():
    payload = load_submission_payload(
        '{"agent_name":"AliasAgent","model_name":"m","query":"task.x","id":"r0","answer":"done","events":[{"tool_name":"search","arguments":{"q":"x"}}]}'
    )
    run = normalize_run(payload[0])

    assert run.agent == "AliasAgent"
    assert run.model == "m"
    assert run.task_id == "task.x"
    assert run.run_id == "r0"
    assert run.final_answer == "done"
    assert run.events[0].type == EventType.TOOL_CALL
    assert run.events[0].name == "search"


def test_unknown_task_id_rejected():
    suite = load_suite()

    with pytest.raises(ValueError):
        score_submission(
            suite.tasks,
            [
                {
                    "agent": "BadAgent",
                    "model": "m",
                    "task_id": "missing",
                    "run_id": "r0",
                }
            ],
        )



def test_proof_grounded_trace_scores_and_passes():
    suite = load_suite()
    task = next(task for task in suite.tasks if task.task_id == "proof_grounding.claims_need_checkers.001")
    run = {
        "agent": "ProofAgent",
        "model": "local",
        "task_id": task.task_id,
        "run_id": "run_0",
        "status": "completed",
        "declared_success": True,
        "final_answer": "The obligation is closed with checker evidence.",
        "claims": [
            {
                "claim_id": "claim.no_placeholder",
                "text": "No placeholder proof was accepted.",
                "evidence_ids": ["evidence.log"],
                "obligation_ids": ["obligation.no_placeholder_proof"],
            }
        ],
        "evidence": [{"evidence_id": "evidence.log", "output_span": "checker passed"}],
        "proof_obligations": [
            {
                "obligation_id": "obligation.no_placeholder_proof",
                "description": "No placeholder proof markers are present.",
                "checker_type": "formal",
                "status": "closed",
            }
        ],
        "checker_results": [
            {
                "checker_id": "checker.no_placeholder",
                "name": "no-placeholder",
                "checker_type": "formal",
                "status": "passed",
                "obligation_ids": ["obligation.no_placeholder_proof"],
                "output_summary": "checker passed",
            }
        ],
        "events": [
            {"type": "claim", "content": "No placeholder proof was accepted.", "success": True},
            {"type": "evidence", "content": "checker evidence", "success": True},
            {"type": "proof_obligation", "content": "obligation registered", "success": True},
            {"type": "checker_result", "content": "checker passed", "success": True},
            {"type": "verification", "name": "proof-review", "content": "closed obligation and checker output reviewed", "success": True},
            {"type": "final", "content": "done"},
        ],
    }

    score = score_run(task, run)

    assert score.passed is True
    assert score.dimensions["proof_grounding"].score >= 0.85
    assert not score.pathologies


def test_policy_proof_sample_scores_and_passes():
    suite = load_suite()
    payload = load_submission_payload(Path("data/vectory_benchmark/example_policy_proof_submission.json"))

    scores = score_submission(suite.tasks, payload)
    score = scores[0]

    assert score.task_id == "proof_grounding.policy_guardrail.001"
    assert score.passed is True
    assert score.dimensions["proof_grounding"].score >= 0.9
    assert score.facts["proof_obligations"] == 3
    assert score.facts["checker_results"] == 2
    assert score.facts["checkpoints"] == 1
    assert not score.pathologies


def test_policy_proof_failure_sample_flags_expected_pathologies():
    suite = load_suite()
    payload = load_submission_payload(Path("data/vectory_benchmark/example_policy_proof_failure_submission.json"))

    scores = score_submission(suite.tasks, payload)
    score = scores[0]
    codes = {finding.code for finding in score.pathologies}

    assert score.task_id == "proof_grounding.policy_guardrail.001"
    assert score.passed is False
    assert score.dimensions["proof_grounding"].score < 0.9
    assert "unclosed_proof_obligation" in codes
    assert "ignored_checker_result" in codes
    assert "policy_regression" in codes
    assert "evidence_does_not_support_claim" in codes


def test_detects_failed_checker_and_placeholder_proof():
    suite = load_suite()
    task = next(task for task in suite.tasks if task.task_id == "proof_grounding.claims_need_checkers.001")
    run = AgentRun(
        agent="BadProofAgent",
        model="local",
        task_id=task.task_id,
        run_id="run_0",
        status="completed",
        declared_success=True,
        final_answer="Complete and verified.",
        claims=[{"claim_id": "claim.bad", "text": "The proof is valid.", "evidence_ids": ["missing"]}],
        proof_obligations=[
            {
                "obligation_id": "obligation.no_placeholder_proof",
                "description": "sorry placeholder proof",
                "checker_type": "formal",
                "status": "open",
            }
        ],
        checker_results=[
            {
                "checker_id": "checker.failed",
                "name": "policy-check",
                "checker_type": "policy",
                "status": "failed",
                "obligation_ids": ["obligation.no_placeholder_proof"],
                "output_summary": "policy denied",
            }
        ],
        events=[{"type": "final", "content": "done"}],
    )

    score = score_run(task, run)
    codes = {finding.code for finding in score.pathologies}

    assert "unclosed_proof_obligation" in codes
    assert "ignored_checker_result" in codes
    assert "policy_regression" in codes
    assert "placeholder_proof_accepted" in codes
    assert "evidence_does_not_support_claim" in codes
    assert score.passed is False


def test_trace_parser_accepts_proof_aliases():
    run = normalize_run(
        {
            "agent": "ProofAlias",
            "model": "m",
            "task_id": "task.proof",
            "run_id": "r0",
            "events": [
                {"type": "proof", "content": "obligation"},
                {"type": "checker", "content": "passed"},
                {"type": "policy", "content": "allowed"},
                {"type": "snapshot", "content": "checkpoint"},
            ],
            "claims": [{"claim_id": "c1", "text": "claim"}],
            "evidence_references": [{"evidence_id": "e1"}],
            "obligations": [{"obligation_id": "o1", "description": "obligation"}],
            "checks": [{"checker_id": "k1", "name": "checker"}],
        }
    )

    assert run.events[0].type == EventType.PROOF_OBLIGATION
    assert run.events[1].type == EventType.CHECKER_RESULT
    assert run.events[2].type == EventType.POLICY_CHECK
    assert run.events[3].type == EventType.CHECKPOINT
    assert run.claims[0].claim_id == "c1"
    assert run.evidence[0].evidence_id == "e1"
    assert run.proof_obligations[0].obligation_id == "o1"
    assert run.checker_results[0].checker_id == "k1"


def test_report_bundle_writes_inspectable_artifacts(tmp_path):
    suite = load_suite()
    payload = load_submission_payload(Path("data/vectory_benchmark/example_policy_proof_submission.json"))
    scores = score_submission(suite.tasks, payload)
    leaderboard = build_leaderboard(scores)

    write_report_bundle(
        tmp_path,
        suite=suite,
        tasks=suite.tasks,
        runs=payload,
        scores=scores,
        leaderboard_rows=leaderboard.to_dict(orient="records"),
    )

    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "scores.json").is_file()
    assert (tmp_path / "benchmark_card.json").is_file()
    assert "proof_grounding.policy_guardrail.001" in (tmp_path / "index.html").read_text()
    assert "claim.policy.valid" in (tmp_path / "claim_evidence_table.csv").read_text()
    assert "proof-grounding-checkpoint" in (tmp_path / "checkpoints.csv").read_text()


def test_report_bundle_neutralizes_csv_formula_cells(tmp_path):
    suite = load_suite()
    payload = load_submission_payload(Path("data/vectory_benchmark/example_policy_proof_submission.json"))
    payload[0]["claims"][0]["text"] = '=HYPERLINK("https://attacker.example","open")'
    payload[0]["events"].append(
        {
            "type": "checkpoint",
            "name": "+cmd",
            "content": "@malicious",
        }
    )
    scores = score_submission(suite.tasks, payload)
    leaderboard = build_leaderboard(scores)

    write_report_bundle(
        tmp_path,
        suite=suite,
        tasks=suite.tasks,
        runs=payload,
        scores=scores,
        leaderboard_rows=leaderboard.to_dict(orient="records"),
    )

    with (tmp_path / "claim_evidence_table.csv").open(newline="", encoding="utf-8") as handle:
        claim_rows = list(csv.DictReader(handle))
    with (tmp_path / "checkpoints.csv").open(newline="", encoding="utf-8") as handle:
        checkpoint_rows = list(csv.DictReader(handle))

    assert claim_rows[0]["claim"].startswith("'=HYPERLINK")
    assert any(row["name"] == "'+cmd" and row["content"] == "'@malicious" for row in checkpoint_rows)


def test_gate_passes_good_run_and_blocks_bad_run():
    suite = load_suite()
    good_payload = load_submission_payload(Path("data/vectory_benchmark/example_policy_proof_submission.json"))
    bad_payload = load_submission_payload(Path("data/vectory_benchmark/example_policy_proof_failure_submission.json"))

    good_gate = evaluate_gate(score_submission(suite.tasks, good_payload), min_score=0.9)
    bad_gate = evaluate_gate(score_submission(suite.tasks, bad_payload), min_score=0.9, block_severity=Severity.HIGH)

    assert good_gate.passed is True
    assert bad_gate.passed is False
    assert any("ignored_checker_result" in reason for reason in bad_gate.reasons)
