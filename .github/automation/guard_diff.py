"""Reject automated changes that leave the scope an unattended agent may touch.

Usage: python .github/automation/guard_diff.py BASE_REF [--allow-roadmap-only]

Checks the working tree plus commits since BASE_REF. Protected paths cover CI,
packaging, dependencies, release metadata, infrastructure, and this automation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from fnmatch import fnmatch

PROTECTED = [
    ".github/*",
    ".agents/*",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements*.txt",
    "MANIFEST.in",
    "Dockerfile",
    "infra/*",
    "LICENSE",
    "CHANGELOG.md",
    "components/__init__.py",
    "vectory_cli/__init__.py",
    ".streamlit/*",
    "*.sh",
]
SOURCE_PREFIXES = ("components/", "vectory_cli/", "pages/", "app.py")
MAX_CHANGED_LINES = 1500


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("--allow-roadmap-only", action="store_true")
    args = parser.parse_args(argv)

    _git("add", "--intent-to-add", "--all")
    numstat = _git("diff", "--numstat", args.base).splitlines()
    problems = []
    changed = []
    total = 0
    for line in numstat:
        added, removed, path = line.split("\t", 2)
        changed.append(path)
        if added == "-":
            problems.append(f"binary file: {path}")
            continue
        total += int(added) + int(removed)
        if any(fnmatch(path, pattern) for pattern in PROTECTED):
            problems.append(f"protected path: {path}")

    if not changed:
        problems.append("no changes")
    if args.allow_roadmap_only:
        problems.extend(f"outside roadmap: {path}" for path in changed if path != "roadmap/queue.json")
    else:
        if total > MAX_CHANGED_LINES:
            problems.append(f"{total} changed lines exceeds {MAX_CHANGED_LINES}")
        touches_source = any(path.startswith(SOURCE_PREFIXES) for path in changed)
        touches_tests = any(path.startswith("tests/") for path in changed)
        if touches_source and not touches_tests:
            problems.append("source changed without test changes")

    for problem in problems:
        print(f"::error::Automation guard: {problem}")
    if not problems:
        print(f"Automation guard passed: {len(changed)} files, {total} changed lines")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
