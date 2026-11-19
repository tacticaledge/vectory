"""
Tests for VectoryBenchmark scoring, pathologies, and aggregation.
"""

import csv
import json
from pathlib import Path
from statistics import mean

import pytest

from components.vectory_benchmark.leaderboard import build_leaderboard
from components.vectory_benchmark.pathology import detect_pathologies
from components.vectory_benchmark.reports import (
    evaluate_gate,
    write_report_bundle,
)
from components.vectory_benchmark.reliability import (
    pass_at_k,
    pass_hat_k,
    reliability_summary,
    task_reliability_rows,
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


def _repeat_task(task_id: str = "coding.repeat") -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        title="Repeat Test",
        domain="coding",
        intent="Fix code after inspection.",
        setup="Inspect, edit, verify.",
        success_criteria=["fixed"],
        checks=TaskChecks(
            expected_answer_keywords=["fix"],
            required_event_types=[EventType.FILE_EDIT, EventType.VERIFICATION],
            reality_sampling_tools=["read_file"],
            min_reality_events=1,
            requires_verification=True,
        ),
        limits=TaskLimits(max_events=20, max_tool_calls=10, turn_cliff_events=12),
    )


def _repeated_scores(
    pass_flags: list[bool],
    task_id: str = "coding.repeat",
    agent: str = "RepeatAgent",
):
    """Score one run per flag so the task has repeated trials with known outcomes."""

    task = _repeat_task(task_id)
    scores = []
    for index, should_pass in enumerate(pass_flags):
        if should_pass:
            run = {
                "agent": agent,
                "model": "local",
                "task_id": task_id,
                "run_id": f"{agent}.{task_id}.run_{index}",
                "status": "completed",
                "final_answer": "Applied the fix and verified the test.",
                "declared_success": True,
                "events": [
                    {"type": "tool_call", "name": "read_file", "path": "src/parser.py", "success": True},
                    {"type": "file_edit", "path": "src/parser.py", "success": True},
                    {"type": "verification", "name": "pytest", "success": True},
                    {"type": "final", "content": "done"},
                ],
            }
        else:
            run = {
                "agent": agent,
                "model": "local",
                "task_id": task_id,
                "run_id": f"{agent}.{task_id}.run_{index}",
                "status": "failed",
                "final_answer": "",
                "events": [{"type": "error", "content": "crashed"}],
            }
        scores.append(score_run(task, run))
    return scores


def test_pass_at_k_and_pass_hat_k_match_hand_computed_cases():
    # n=8, c=6, k=4: C(2,4)=0 so pass@4 is certain, and pass^4 is C(6,4)/C(8,4)=15/70.
    assert pass_at_k(8, 6, 4) == pytest.approx(1.0)
    assert pass_hat_k(8, 6, 4) == pytest.approx(15 / 70)
    # n=5, c=2, k=2: pass@2 = 1 - C(3,2)/C(5,2), pass^2 = C(2,2)/C(5,2).
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert pass_hat_k(5, 2, 2) == pytest.approx(0.1)
    # k=1 collapses to the observed success rate for both estimators.
    assert pass_at_k(4, 3, 1) == pytest.approx(0.75)
    assert pass_hat_k(4, 3, 1) == pytest.approx(0.75)
    assert pass_at_k(3, 0, 2) == pytest.approx(0.0)
    assert pass_hat_k(3, 3, 3) == pytest.approx(1.0)


def test_pass_at_k_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        pass_at_k(3, 4, 2)
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 4)
    with pytest.raises(ValueError):
        pass_hat_k(0, 0, 1)
    with pytest.raises(ValueError):
        pass_hat_k(3, 1, 0)


def test_leaderboard_reports_pass_at_k_for_repeated_runs():
    scores = _repeated_scores([True, True, False, True])

    assert [score.passed for score in scores] == [True, True, False, True]

    rows = task_reliability_rows(scores)
    leaderboard = build_leaderboard(scores)
    entry = leaderboard.iloc[0]

    assert rows == [
        {
            "agent": "RepeatAgent",
            "model": "local",
            "task_id": "coding.repeat",
            "domain": "coding",
            "trials": 4,
            "successes": 3,
            "k": 4,
            "pass_at_k": 1.0,
            "pass_hat_k": 0.0,
        }
    ]
    assert entry["reliability_k"] == 4
    assert entry["pass_at_k"] == pytest.approx(1.0)
    assert entry["pass_hat_k"] == pytest.approx(0.0)
    assert entry["reliability_tasks"] == 1
    assert entry["robust_pass_at_5"] == pytest.approx(1.0)
    assert entry["pass_at_1"] == pytest.approx(0.75)


def test_reliability_summary_uses_k_supported_by_every_repeated_task():
    scores = _repeated_scores([True, False, False], task_id="coding.repeat_a")
    scores += _repeated_scores([True, True], task_id="coding.repeat_b")
    scores += _repeated_scores([False], task_id="coding.repeat_single")

    summary = reliability_summary(scores)
    rows = {row["task_id"]: row for row in task_reliability_rows(scores)}

    assert summary["reliability_k"] == 2
    assert summary["reliability_tasks"] == 2
    # pass@2 is 1 - C(2,2)/C(3,2) for the 1-of-3 task and 1.0 for the 2-of-2 task.
    assert summary["pass_at_k"] == pytest.approx(mean([2 / 3, 1.0]), abs=1e-4)
    assert summary["pass_hat_k"] == pytest.approx(mean([0.0, 1.0]), abs=1e-4)
    assert rows["coding.repeat_single"]["k"] is None
    assert rows["coding.repeat_single"]["pass_at_k"] is None
    assert reliability_summary(scores, k=3)["reliability_tasks"] == 1


def test_mixed_agent_reliability_uses_one_shared_k():
    scores = _repeated_scores([True, True, False, True], agent="DeepAgent")
    scores += _repeated_scores([True, False], agent="FastAgent")
    scores += _repeated_scores([True], agent="SingleRunAgent")

    summary = reliability_summary(scores)
    rows = {row["agent"]: row for row in task_reliability_rows(scores)}
    leaderboard = build_leaderboard(scores)
    entries = {entry["agent"]: entry for entry in leaderboard.to_dict(orient="records")}

    # The smallest repeated trial count in the submission is 2, so every task row,
    # every leaderboard entry, and the overall summary must report k=2.
    assert summary["reliability_k"] == 2
    assert {row["k"] for row in rows.values()} == {2, None}
    assert rows["DeepAgent"]["k"] == 2
    assert rows["FastAgent"]["k"] == 2
    assert rows["SingleRunAgent"]["k"] is None
    assert entries["DeepAgent"]["reliability_k"] == 2
    assert entries["FastAgent"]["reliability_k"] == 2
    assert entries["SingleRunAgent"]["reliability_k"] is None
    # pass^2 is C(3,2)/C(4,2) for the 3-of-4 agent and C(1,2)/C(2,2) for the 1-of-2 agent.
    assert rows["DeepAgent"]["pass_hat_k"] == pytest.approx(0.5)
    assert rows["FastAgent"]["pass_hat_k"] == pytest.approx(0.0)
    assert entries["DeepAgent"]["pass_hat_k"] == pytest.approx(0.5)
    assert entries["FastAgent"]["pass_hat_k"] == pytest.approx(0.0)
    assert entries["SingleRunAgent"]["pass_hat_k"] is None
    assert summary["reliability_tasks"] == 2
    assert summary["pass_hat_k"] == pytest.approx(mean([0.5, 0.0]), abs=1e-4)


def test_leaderboard_keeps_robust_pass_at_5_without_repeated_runs():
    scores = _repeated_scores([True])

    leaderboard = build_leaderboard(scores)
    entry = leaderboard.iloc[0]

    assert entry["robust_pass_at_5"] == pytest.approx(1.0)
    assert entry["reliability_k"] is None
    assert entry["pass_at_k"] is None
    assert entry["pass_hat_k"] is None
    assert reliability_summary(scores)["reliability_k"] is None


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


def test_report_bundle_includes_task_reliability(tmp_path):
    suite = load_suite()
    scores = _repeated_scores([True, False, True])
    leaderboard = build_leaderboard(scores)

    write_report_bundle(
        tmp_path,
        suite=suite,
        tasks=suite.tasks,
        runs=[],
        scores=scores,
        leaderboard_rows=leaderboard.to_dict(orient="records"),
    )

    reliability_rows = json.loads((tmp_path / "task_reliability.json").read_text())
    report = (tmp_path / "index.html").read_text()

    assert reliability_rows[0]["task_id"] == "coding.repeat"
    assert reliability_rows[0]["trials"] == 3
    assert reliability_rows[0]["successes"] == 2
    assert reliability_rows[0]["pass_at_k"] == pytest.approx(1.0)
    assert reliability_rows[0]["pass_hat_k"] == pytest.approx(0.0)
    assert "pass_hat_k" in (tmp_path / "task_reliability.csv").read_text()
    assert "Task Reliability" in report
    assert "Pass^3" in report


def test_mixed_submission_leaderboard_json_is_strict(tmp_path):
    suite = load_suite()
    scores = _repeated_scores([True, False], agent="RepeatAgent")
    scores += _repeated_scores([True], agent="SingleRunAgent")
    leaderboard_rows = build_leaderboard(scores).to_dict(orient="records")

    # allow_nan=False stands in for a strict consumer: it refuses to emit NaN.
    assert "NaN" not in json.dumps(leaderboard_rows, allow_nan=False)

    write_report_bundle(
        tmp_path,
        suite=suite,
        tasks=suite.tasks,
        runs=[],
        scores=scores,
        leaderboard_rows=leaderboard_rows,
    )

    written = (tmp_path / "leaderboard.json").read_text()
    entries = {row["agent"]: row for row in json.loads(written)}

    assert "NaN" not in written
    assert entries["RepeatAgent"]["reliability_k"] == 2
    assert entries["SingleRunAgent"]["reliability_k"] is None
    assert entries["SingleRunAgent"]["pass_at_k"] is None
    assert entries["SingleRunAgent"]["pass_hat_k"] is None
    assert entries["SingleRunAgent"]["reliability_tasks"] is None
    assert entries["SingleRunAgent"]["pass_at_1"] == pytest.approx(1.0)


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
