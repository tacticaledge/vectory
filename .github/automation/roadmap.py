"""Manage the automated feature queue in roadmap/queue.json.

Commands:
  next                         Print the highest-priority queued item as JSON.
  set-status ID STATUS         Update one item's status.
  merge-research RESEARCH      Add validated, deduplicated research proposals.
  schema                       Print the JSON schema for research output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

QUEUE = Path(__file__).resolve().parents[2] / "roadmap" / "queue.json"
STATUSES = {"queued", "shipped", "blocked"}
MAX_NEW_ITEMS = 3
MAX_QUEUED = 25
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

RESEARCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["proposals"],
    "properties": {
        "proposals": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "title", "summary", "acceptance", "sources", "priority"],
                "properties": {
                    "id": {"type": "string", "pattern": SLUG.pattern, "maxLength": 60},
                    "title": {"type": "string", "minLength": 10, "maxLength": 90},
                    "summary": {"type": "string", "minLength": 80, "maxLength": 900},
                    "acceptance": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {"type": "string", "minLength": 10, "maxLength": 300},
                    },
                    "sources": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {"type": "string", "pattern": "^https://", "maxLength": 300},
                    },
                    "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                },
            },
        }
    },
}


def load() -> dict:
    return json.loads(QUEUE.read_text(encoding="utf-8"))


def save(queue: dict) -> None:
    QUEUE.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def next_item(queue: dict) -> dict | None:
    queued = [item for item in queue["items"] if item["status"] == "queued"]
    return min(queued, key=lambda item: (item["priority"], item["added"]), default=None)


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _validate_proposal(proposal: dict) -> str | None:
    """Return a rejection reason, or None when the proposal fits the schema."""
    required = RESEARCH_SCHEMA["properties"]["proposals"]["items"]["required"]
    if set(proposal) != set(required):
        return "unexpected or missing fields"
    if not SLUG.match(proposal["id"]) or len(proposal["id"]) > 60:
        return "invalid id"
    if not 10 <= len(proposal["title"]) <= 90 or not 80 <= len(proposal["summary"]) <= 900:
        return "title or summary length out of range"
    if not 2 <= len(proposal["acceptance"]) <= 6:
        return "acceptance criteria count out of range"
    if not all(isinstance(line, str) and 10 <= len(line) <= 300 for line in proposal["acceptance"]):
        return "acceptance criterion length out of range"
    if not 1 <= len(proposal["sources"]) <= 5:
        return "source count out of range"
    if not all(isinstance(url, str) and url.startswith("https://") and len(url) <= 300 for url in proposal["sources"]):
        return "sources must be HTTPS URLs"
    if not isinstance(proposal["priority"], int) or not 1 <= proposal["priority"] <= 5:
        return "priority out of range"
    return None


def merge_research(queue: dict, research: dict) -> list[str]:
    known_ids = {item["id"] for item in queue["items"]}
    known_titles = {_normalized(item["title"]) for item in queue["items"]}
    queued = sum(item["status"] == "queued" for item in queue["items"])
    added = []
    for proposal in sorted(research.get("proposals", []), key=lambda item: item.get("priority", 5)):
        if len(added) >= MAX_NEW_ITEMS or queued >= MAX_QUEUED:
            break
        reason = _validate_proposal(proposal)
        if reason is None and (proposal["id"] in known_ids or _normalized(proposal["title"]) in known_titles):
            reason = "duplicate"
        if reason is not None:
            print(f"Skipped proposal {proposal.get('id')!r}: {reason}", file=sys.stderr)
            continue
        queue["items"].append(
            {
                **proposal,
                "origin": "research",
                "status": "queued",
                "added": date.today().isoformat(),
            }
        )
        known_ids.add(proposal["id"])
        known_titles.add(_normalized(proposal["title"]))
        queued += 1
        added.append(proposal["id"])
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("next")
    status_parser = commands.add_parser("set-status")
    status_parser.add_argument("item_id")
    status_parser.add_argument("status", choices=sorted(STATUSES))
    research_parser = commands.add_parser("merge-research")
    research_parser.add_argument("research", type=Path)
    commands.add_parser("schema")
    args = parser.parse_args(argv)

    if args.command == "schema":
        print(json.dumps(RESEARCH_SCHEMA))
        return 0
    queue = load()
    if args.command == "next":
        item = next_item(queue)
        if item is None:
            return 1
        print(json.dumps(item, indent=2, ensure_ascii=False))
        return 0
    if args.command == "set-status":
        matches = [item for item in queue["items"] if item["id"] == args.item_id]
        if not matches:
            parser.error(f"Unknown roadmap item: {args.item_id}")
        matches[0]["status"] = args.status
        save(queue)
        return 0
    payload = json.loads(args.research.read_text(encoding="utf-8"))
    research = payload.get("structured_output", payload)
    added = merge_research(queue, research)
    save(queue)
    print(json.dumps({"added": added}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
