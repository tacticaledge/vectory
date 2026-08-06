"""Optional trusted formal checker execution for Vectory Benchmark."""

from __future__ import annotations

import subprocess
from pathlib import Path

from components.vectory_benchmark.schemas import AgentRun, BenchmarkTask, CheckerResult


def _summarize_output(stdout: str, stderr: str, limit: int = 1200) -> str:
    text = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
    if not text:
        return "checker produced no output"
    return text[:limit]


def run_trusted_formal_checkers(task: BenchmarkTask, run: AgentRun, workspace: Path) -> AgentRun:
    """Run suite-defined checker commands and append their results to a copy of the run.

    Commands are read only from the trusted local suite manifest. The submitted run
    may contain checker results, but it cannot provide executable commands.
    """
    if not task.checks.formal_checkers:
        return run

    root = workspace.resolve()
    if not root.is_dir():
        raise ValueError(f"Formal runtime workspace does not exist: {workspace}")

    enriched = run.model_copy(deep=True)
    for index, checker in enumerate(task.checks.formal_checkers):
        try:
            completed = subprocess.run(
                checker.command,
                cwd=str(root),
                check=False,
                capture_output=True,
                text=True,
                timeout=checker.timeout_seconds,
            )
            status = "passed" if completed.returncode == 0 else "failed"
            output_summary = _summarize_output(completed.stdout, completed.stderr)
        except FileNotFoundError as exc:
            status = "error"
            output_summary = f"checker executable not found: {exc.filename}"
        except subprocess.TimeoutExpired as exc:
            status = "error"
            output_summary = f"checker timed out after {checker.timeout_seconds}s: {exc}"

        enriched.checker_results.append(
            CheckerResult(
                checker_id=f"trusted.{checker.name}.{index}",
                name=checker.name,
                checker_type=checker.checker_type,
                status=status,
                obligation_ids=checker.obligation_ids,
                output_summary=output_summary,
                command=checker.command,
            )
        )

    return enriched
