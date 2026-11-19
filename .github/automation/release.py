"""Prepare a release from conventional commits since the latest version tag.

Usage: python .github/automation/release.py

Prints the new version, or nothing when no user-facing change shipped. Updates
components/__init__.py and prepends the release notes to CHANGELOG.md.
"""

from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "components" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"
SECTIONS = {"feat": "Features", "fix": "Fixes", "perf": "Performance", "docs": "Documentation"}
COMMIT = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?!?: (?P<subject>.+)$")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def main() -> int:
    tag = _git("describe", "--tags", "--abbrev=0", "--match", "v*.*.*").strip()
    subjects = _git("log", "--no-merges", "--format=%s", f"{tag}..HEAD").splitlines()
    notes: dict[str, list[str]] = {}
    for subject in subjects:
        match = COMMIT.match(subject)
        if match and match["type"] in SECTIONS:
            notes.setdefault(SECTIONS[match["type"]], []).append(match["subject"])
    if not any(section in notes for section in ("Features", "Fixes", "Performance")):
        return 0

    source = VERSION_FILE.read_text(encoding="utf-8")
    current = re.search(r'__version__ = "(\d+)\.(\d+)\.(\d+)"', source)
    if current is None or f"v{'.'.join(current.groups())}" != tag:
        raise SystemExit(f"components/__init__.py version does not match latest tag {tag}")
    major, minor, patch = (int(part) for part in current.groups())
    version = f"{major}.{minor + 1}.0" if "Features" in notes else f"{major}.{minor}.{patch + 1}"
    VERSION_FILE.write_text(source.replace(current.group(0), f'__version__ = "{version}"'), encoding="utf-8")

    entry = [f"## {version} - {date.today().isoformat()}", ""]
    for section in SECTIONS.values():
        if section in notes:
            entry += [f"### {section}", ""] + [f"- {line}" for line in notes[section]] + [""]
    existing = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.exists() else "# Changelog\n\n"
    header, _, body = existing.partition("\n\n")
    CHANGELOG.write_text(f"{header}\n\n" + "\n".join(entry) + "\n" + body, encoding="utf-8")
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
