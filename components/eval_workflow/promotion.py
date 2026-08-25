"""Promote accepted failure modes into inspectable evaluator definitions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "failure-mode"


def _example_texts(failure_mode: Mapping[str, Any]) -> list[str]:
    examples = failure_mode.get("examples") or failure_mode.get("example_quotes") or []
    if isinstance(examples, str):
        examples = [examples]
    return [str(example).strip() for example in examples if str(example).strip()]


def build_evaluator_definition(
    name: str,
    failure_mode: Mapping[str, Any],
    *,
    kind: str = "llm_judge",
    input_fields: Sequence[str] | None = None,
    pass_definition: str | None = None,
    rule_pattern: str | None = None,
    source_annotation_ids: Sequence[int | str] | None = None,
    expert_examples: Sequence[Mapping[str, Any]] | None = None,
    errors_reviewed: bool = False,
) -> dict[str, Any]:
    """Build an unvalidated evaluator artifact from a human-accepted failure mode."""
    description = str(failure_mode.get("description") or failure_mode.get("definition") or "").strip()
    if not description:
        raise ValueError("Failure mode must have a description before promotion")
    if kind not in {"llm_judge", "regex", "contains"}:
        raise ValueError(f"Unsupported evaluator kind: {kind}")
    if kind in {"regex", "contains"} and not rule_pattern:
        raise ValueError(f"{kind} evaluators require a rule pattern")

    evaluator_id = f"failure-{_slugify(name)}"
    examples = _example_texts(failure_mode)
    fields = list(input_fields or ["input", "output", "reference", "trace"])
    pass_text = pass_definition or (
        f"The failure mode '{name}' is not present. The output satisfies the relevant requirement."
    )
    fail_text = description
    normalized_expert_examples = []
    for example in expert_examples or []:
        critique = str(example.get("critique") or "").strip()
        result = str(example.get("result") or "").strip().casefold()
        if not critique:
            raise ValueError("Every expert example must include a detailed critique")
        if result not in {"pass", "fail"}:
            raise ValueError("Every expert example result must be Pass or Fail")
        normalized_expert_examples.append(
            {
                "input": dict(example.get("input") or {}),
                "critique": critique,
                "result": result.title(),
            }
        )

    definition: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluator_id": evaluator_id,
        "name": name,
        "kind": kind,
        "status": "draft_unvalidated",
        "input_fields": fields,
        "failure_mode": {
            "name": name,
            "description": description,
            "source_examples": examples,
            "source_annotation_ids": [str(value) for value in (source_annotation_ids or [])],
        },
        "decision": {
            "type": "binary",
            "pass_definition": pass_text,
            "fail_definition": fail_text,
        },
        "validation": {
            "required": True,
            "validated": False,
            "minimum_tpr": 0.8,
            "minimum_tnr": 0.8,
            "note": "Do not use as a release gate until measured against held-out human labels.",
        },
        "lifecycle_checkpoints": {
            "human_review_completed": bool(source_annotation_ids),
            "obvious_errors_reviewed": bool(errors_reviewed),
            "expert_examples_include_pass_and_fail": {
                example["result"] for example in normalized_expert_examples
            } == {"Pass", "Fail"},
            "note": (
                "Fix pervasive product errors and repeat human review before investing in automation."
            ),
        },
        "expert_examples": normalized_expert_examples,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if kind == "llm_judge":
        rendered_expert_examples = []
        for index, example in enumerate(normalized_expert_examples[:8], start=1):
            rendered_expert_examples.append(
                f"<example-{index}>\n"
                f"<input>{json.dumps(example['input'], ensure_ascii=False, default=str)}</input>\n"
                f"<critique>{example['critique']}</critique>\n"
                f"<result>{example['result']}</result>\n"
                f"</example-{index}>"
            )
        example_section = "\n\n".join(rendered_expert_examples)
        if not example_section:
            example_section = "No expert examples attached yet. Add both Pass and Fail critiques before validation."
        definition["prompt"] = f"""You are evaluating one specific failure mode: {name}.

## Definitions
PASS: {pass_text}
FAIL: {fail_text}

## Principal domain expert examples
{example_section}

## Trace or output to evaluate
{{{{evaluation_input}}}}

Assess only this failure mode. Cite concrete evidence from the supplied input in the critique.
Return JSON that conforms to the required schema. Write the critique before the result."""
        definition["output_schema"] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["critique", "result"],
            "properties": {
                "critique": {"type": "string"},
                "result": {"type": "string", "enum": ["Pass", "Fail"]},
            },
        }
    else:
        definition["rule"] = {
            "pattern": rule_pattern,
            "match_means": "Fail",
            "case_sensitive": False,
        }
    return definition
