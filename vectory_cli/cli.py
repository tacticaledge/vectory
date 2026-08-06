"""Vectory command line interface."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

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
        description="Run Vectory locally and score Vectory Benchmark submissions.",
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

    doctor_parser = subparsers.add_parser("doctor", help="Check local Vectory installation health.")
    doctor_parser.set_defaults(func=doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
