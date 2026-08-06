"""Leaderboard aggregation for VectoryBenchmark score reports."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

import pandas as pd

from components.vectory_benchmark.schemas import LeaderboardEntry, RunScore


def _robust_pass_at_5(scores: list[RunScore]) -> float:
    by_task: dict[str, list[RunScore]] = defaultdict(list)
    for score in scores:
        by_task[score.task_id].append(score)
    if not by_task:
        return 0.0
    task_rates = []
    for task_scores in by_task.values():
        first_five = task_scores[:5]
        task_rates.append(1.0 if any(score.passed for score in first_five) else 0.0)
    return mean(task_rates)


def build_leaderboard(scores: list[RunScore]) -> pd.DataFrame:
    """Aggregate run scores into a ranked leaderboard DataFrame."""
    grouped: dict[tuple[str, str], list[RunScore]] = defaultdict(list)
    for score in scores:
        grouped[(score.agent, score.model)].append(score)

    entries: list[LeaderboardEntry] = []
    for (agent, model), group in grouped.items():
        vectory_score = mean(score.vectory_score for score in group)
        pass_at_1 = mean(1.0 if score.passed else 0.0 for score in group)
        productive_work_ratio = mean(
            float(score.facts.get("productive_work_ratio", score.dimensions["trace_productivity"].score))
            for score in group
        )
        pathology_risk = mean(
            min(1.0, sum(finding.score_penalty for finding in score.pathologies))
            for score in group
        )
        agent_control_index = mean(
            float(score.facts.get("agent_control_index", score.dimensions["safety_control"].score))
            if "safety_control" in score.dimensions
            else float(score.facts.get("agent_control_index", score.dimensions["agent_control"].score))
            for score in group
        )
        entries.append(
            LeaderboardEntry(
                rank=0,
                agent=agent,
                model=model,
                runs=len(group),
                tasks=len({score.task_id for score in group}),
                vectory_score=round(vectory_score, 4),
                pass_at_1=round(pass_at_1, 4),
                robust_pass_at_5=round(_robust_pass_at_5(group), 4),
                productive_work_ratio=round(productive_work_ratio, 4),
                pathology_risk=round(pathology_risk, 4),
                agent_control_index=round(agent_control_index, 4),
            )
        )

    entries.sort(
        key=lambda entry: (
            entry.vectory_score,
            entry.agent_control_index,
            entry.productive_work_ratio,
            -entry.pathology_risk,
        ),
        reverse=True,
    )
    for index, entry in enumerate(entries, start=1):
        entry.rank = index

    return pd.DataFrame([entry.model_dump() for entry in entries])
