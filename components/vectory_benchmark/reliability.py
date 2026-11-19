"""Repeated-run reliability estimators for VectoryBenchmark."""

from __future__ import annotations

from collections import defaultdict
from math import comb
from statistics import mean
from typing import Any, Iterable

from components.vectory_benchmark.schemas import RunScore


def _validate_trials(trials: int, successes: int, k: int) -> None:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    if not 1 <= k <= trials:
        raise ValueError("k must be between 1 and trials")


def pass_at_k(trials: int, successes: int, k: int) -> float:
    """Unbiased pass@k: chance that at least one of k sampled runs passes."""

    _validate_trials(trials, successes, k)
    return 1.0 - comb(trials - successes, k) / comb(trials, k)


def pass_hat_k(trials: int, successes: int, k: int) -> float:
    """Unbiased pass^k: chance that every one of k sampled runs passes."""

    _validate_trials(trials, successes, k)
    return comb(successes, k) / comb(trials, k)


def reliability_k(trial_counts: Iterable[int]) -> int | None:
    """Largest k every repeated-run task supports, or None without repeated runs."""

    repeated = [count for count in trial_counts if count >= 2]
    if not repeated:
        return None
    return min(repeated)


def _task_trials(scores: list[RunScore]) -> dict[tuple[str, str, str], list[RunScore]]:
    grouped: dict[tuple[str, str, str], list[RunScore]] = defaultdict(list)
    for score in scores:
        grouped[(score.agent, score.model, score.task_id)].append(score)
    return grouped


def submission_reliability_k(scores: list[RunScore]) -> int | None:
    """Shared k for a submission: the largest k every repeated-run task supports.

    One k is used for every agent, model, and task so that per-task rows,
    leaderboard entries, and aggregate summaries stay comparable.
    """

    return reliability_k(len(task_scores) for task_scores in _task_trials(scores).values())


def task_reliability_rows(scores: list[RunScore], *, k: int | None = None) -> list[dict[str, Any]]:
    """Build per-task pass@k and pass^k rows, one row per agent/model/task."""

    grouped = _task_trials(scores)
    chosen = k if k is not None else submission_reliability_k(scores)

    rows: list[dict[str, Any]] = []
    for (agent, model, task_id), task_scores in grouped.items():
        trials = len(task_scores)
        successes = sum(1 for score in task_scores if score.passed)
        usable = chosen is not None and chosen <= trials
        rows.append(
            {
                "agent": agent,
                "model": model,
                "task_id": task_id,
                "domain": task_scores[0].domain.value,
                "trials": trials,
                "successes": successes,
                "k": chosen if usable else None,
                "pass_at_k": round(pass_at_k(trials, successes, chosen), 4) if usable else None,
                "pass_hat_k": round(pass_hat_k(trials, successes, chosen), 4) if usable else None,
            }
        )
    rows.sort(key=lambda row: (row["agent"], row["model"], row["task_id"]))
    return rows


def reliability_summary(scores: list[RunScore], *, k: int | None = None) -> dict[str, Any]:
    """Average pass@k and pass^k over the tasks that support the chosen k."""

    counts = [
        (len(task_scores), sum(1 for score in task_scores if score.passed))
        for task_scores in _task_trials(scores).values()
    ]
    chosen = k if k is not None else submission_reliability_k(scores)
    eligible = (
        [(trials, successes) for trials, successes in counts if trials >= chosen]
        if chosen is not None
        else []
    )
    if not eligible:
        return {
            "reliability_k": None,
            "pass_at_k": None,
            "pass_hat_k": None,
            "reliability_tasks": None,
        }
    return {
        "reliability_k": chosen,
        "pass_at_k": round(mean(pass_at_k(trials, successes, chosen) for trials, successes in eligible), 4),
        "pass_hat_k": round(mean(pass_hat_k(trials, successes, chosen) for trials, successes in eligible), 4),
        "reliability_tasks": len(eligible),
    }
