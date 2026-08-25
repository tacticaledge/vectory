"""Normalize common trace shapes into safe, renderer-neutral review segments."""

from __future__ import annotations

import json
from typing import Any, Mapping


_TRACE_FIELDS = ("events", "trace", "trajectory", "messages", "spans", "steps")


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _segment_from_item(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {
            "index": index,
            "kind": "content",
            "role": "unknown",
            "title": f"Step {index + 1}",
            "content": str(item),
            "details": None,
        }
    raw = dict(item)
    role = str(raw.get("role") or raw.get("type") or raw.get("author") or "unknown")
    kind = str(raw.get("type") or role).casefold().replace(" ", "_")
    title = str(raw.get("name") or raw.get("tool_name") or raw.get("title") or role.replace("_", " ").title())
    content = raw.get("content")
    if content is None:
        content = raw.get("text")
    if content is None:
        content = raw.get("message")
    if content is None and kind in {"tool_result", "result", "observation"}:
        content = raw.get("output")
    details = {
        key: value
        for key, value in raw.items()
        if key not in {"role", "type", "author", "name", "tool_name", "title", "content", "text", "message"}
    }
    if content is None and "output" in details:
        content = details.pop("output")
    if isinstance(content, (dict, list)):
        rendered_content = json.dumps(content, indent=2, ensure_ascii=False, default=str)
    else:
        rendered_content = "" if content is None else str(content)
    return {
        "index": index,
        "kind": kind,
        "role": role,
        "title": title,
        "content": rendered_content,
        "details": details or None,
    }


def normalize_trace_segments(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return ordered segments from a common agent/chat trace field, if present."""
    for field in _TRACE_FIELDS:
        if field not in record:
            continue
        value = _maybe_json(record[field])
        if isinstance(value, Mapping):
            for nested_field in _TRACE_FIELDS:
                if isinstance(value.get(nested_field), list):
                    value = value[nested_field]
                    break
        if isinstance(value, list):
            return [_segment_from_item(item, index) for index, item in enumerate(value)]
    return []
