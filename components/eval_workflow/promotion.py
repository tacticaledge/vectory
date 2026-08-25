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


def validate_evaluator_definition(
    value: Mapping[str, Any], *, required_kind: str | None = None
) -> dict[str, Any]:
    """Validate the complete portable evaluator contract at an import boundary."""
    if not isinstance(value, Mapping):
        raise ValueError("Evaluator definition must be a JSON object")
    evaluator = dict(value)
    required_strings = ("schema_version", "evaluator_id", "name", "kind", "status")
    for field in required_strings:
        if not isinstance(evaluator.get(field), str) or not evaluator[field].strip():
            raise ValueError(f"Evaluator definition requires a non-empty {field}")
    if evaluator["schema_version"] != "1.0":
        raise ValueError(f"Unsupported evaluator schema_version: {evaluator['schema_version']}")
    evaluator_id = evaluator["evaluator_id"]
    if any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in evaluator_id):
        raise ValueError("Evaluator definition must have a safe evaluator_id")
    if evaluator["kind"] not in {"llm_judge", "regex", "contains"}:
        raise ValueError(f"Unsupported evaluator kind: {evaluator['kind']}")
    if required_kind and evaluator["kind"] != required_kind:
        raise ValueError(f"Evaluator kind must be {required_kind}")
    if evaluator["status"] not in {"draft_unvalidated", "validated", "validation_failed"}:
        raise ValueError(f"Unsupported evaluator status: {evaluator['status']}")

    input_fields = evaluator.get("input_fields")
    if not isinstance(input_fields, list) or not input_fields or not all(
        isinstance(field, str) and field.strip() for field in input_fields
    ):
        raise ValueError("Evaluator input_fields must be a non-empty list of strings")
    decision = evaluator.get("decision")
    if not isinstance(decision, Mapping) or decision.get("type") != "binary":
        raise ValueError("Evaluator decision.type must be binary")
    for field in ("pass_definition", "fail_definition"):
        if not isinstance(decision.get(field), str) or not decision[field].strip():
            raise ValueError(f"Evaluator decision requires a non-empty {field}")
    if not isinstance(evaluator.get("validation"), Mapping):
        raise ValueError("Evaluator definition requires validation metadata")
    lifecycle = evaluator.get("lifecycle_checkpoints")
    if not isinstance(lifecycle, Mapping):
        raise ValueError("Evaluator definition requires lifecycle checkpoints")
    for checkpoint in (
        "human_review_completed",
        "obvious_errors_reviewed",
        "expert_examples_include_pass_and_fail",
    ):
        if not isinstance(lifecycle.get(checkpoint), bool):
            raise ValueError(f"Evaluator lifecycle checkpoint {checkpoint} must be boolean")

    examples = evaluator.get("expert_examples")
    if not isinstance(examples, list):
        raise ValueError("Evaluator expert_examples must be a list")
    for example in examples:
        if not isinstance(example, Mapping) or not isinstance(example.get("input"), Mapping):
            raise ValueError("Every expert example requires an input object")
        if not isinstance(example.get("critique"), str) or not example["critique"].strip():
            raise ValueError("Every expert example requires a non-empty critique")
        if example.get("result") not in {"Pass", "Fail"}:
            raise ValueError("Every expert example result must be Pass or Fail")

    if evaluator["kind"] == "llm_judge":
        prompt = evaluator.get("prompt")
        if not isinstance(prompt, str) or prompt.count("{{evaluation_input}}") != 1:
            raise ValueError("LLM judge prompt must contain exactly one {{evaluation_input}} placeholder")
        output_schema = evaluator.get("output_schema")
        expected_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["critique", "result"],
            "properties": {
                "critique": {"type": "string", "minLength": 1},
                "result": {"type": "string", "enum": ["Pass", "Fail"]},
            },
        }
        if output_schema != expected_schema:
            raise ValueError("LLM judge output_schema must match Vectory's strict binary contract")
    else:
        rule = evaluator.get("rule")
        if not isinstance(rule, Mapping) or not isinstance(rule.get("pattern"), str):
            raise ValueError("Rule evaluator definition requires a string pattern")
    return evaluator


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
                "critique": {"type": "string", "minLength": 1},
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
