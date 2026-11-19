"""Detect held-out record text that leaked into an evaluator definition."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .schemas import EvaluatorDefinition, parse_evaluator_definition


HELD_OUT_SPLITS = ("dev", "test")
DEFAULT_MINIMUM_CHARACTERS = 8
_WHITESPACE = re.compile(r"\s+")
_ESCAPED_WHITESPACE = re.compile(r"\\[nrtf]|\\u000[bB]")


def normalize_leakage_text(value: Any) -> str:
    """Lowercase and collapse whitespace so formatting alone cannot hide a reused record.

    Serializing a structure escapes the newlines and tabs inside it, so those escapes
    become whitespace again before collapsing. Otherwise a multiline record field would
    never match the serialized expert example that embeds it.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = _ESCAPED_WHITESPACE.sub(
            " ", json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        )
    return _WHITESPACE.sub(" ", text).strip().casefold()


def _search_variants(text: str) -> tuple[str, ...]:
    """List the normalized forms a reused record text can take inside a definition.

    Expert example inputs are stored as mappings and rendered into the judge prompt as
    JSON, so a record's quotes and backslashes appear escaped there while the record
    itself carries them verbatim. Searching the escaped form too keeps that escaping
    from hiding a reused record.
    """
    escaped = json.dumps(text, ensure_ascii=False)[1:-1]
    return (text,) if escaped == text else (text, escaped)


def _definition_sections(evaluator: EvaluatorDefinition) -> list[tuple[str, str]]:
    """List the searchable definition text, most specific location first."""
    sections: list[tuple[str, str]] = []
    for index, example in enumerate(evaluator.expert_examples, start=1):
        sections.append((f"expert_examples[{index}].input", normalize_leakage_text(example.input)))
        sections.append((f"expert_examples[{index}].critique", normalize_leakage_text(example.critique)))
    for index, quote in enumerate(evaluator.failure_mode.source_examples, start=1):
        sections.append((f"failure_mode.source_examples[{index}]", normalize_leakage_text(quote)))
    if evaluator.rule is not None:
        sections.append(("rule.pattern", normalize_leakage_text(evaluator.rule.pattern)))
    sections.extend(
        [
            ("decision.pass_definition", normalize_leakage_text(evaluator.decision.pass_definition)),
            ("decision.fail_definition", normalize_leakage_text(evaluator.decision.fail_definition)),
            ("prompt", normalize_leakage_text(evaluator.prompt)),
        ]
    )
    return [(location, text) for location, text in sections if text]


def detect_evaluator_leakage(
    evaluator: Mapping[str, Any] | EvaluatorDefinition,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    text_fields: Sequence[str] | None = None,
    minimum_characters: int = DEFAULT_MINIMUM_CHARACTERS,
) -> dict[str, Any]:
    """Report every held-out record whose normalized text appears in an evaluator definition.

    Record text is compared against the prompt, expert examples, failure-mode quotes,
    decision definitions, and rule pattern after lowercasing and collapsing whitespace.
    Each record text is searched both verbatim and in its JSON-escaped form, so quotes
    and backslashes cannot hide inside a serialized example or prompt.
    Texts shorter than ``minimum_characters`` are skipped, because incidental short
    phrases match instruction wording without indicating leakage. Only the first
    matching location per record field is reported, so one reused record stays one
    finding. A report is clean only when no record matched *and* at least one record
    text was actually compared, so neither a mismatched field configuration nor records
    whose every value is too short can pass as a completed check.
    """
    if minimum_characters < 1:
        raise ValueError("Minimum leakage comparison length must be positive")
    definition = parse_evaluator_definition(evaluator)
    fields = list(dict.fromkeys(text_fields or definition.input_fields))
    if not fields:
        raise ValueError("Leakage detection requires at least one record field to compare")
    sections = _definition_sections(definition)

    matches: list[dict[str, Any]] = []
    records_with_field = {field: 0 for field in fields}
    compared_texts = {field: 0 for field in fields}
    skipped_short = 0
    for split_name, records in splits.items():
        for index, record in enumerate(records):
            for field in fields:
                if field not in record:
                    continue
                records_with_field[field] += 1
                text = normalize_leakage_text(record[field])
                if len(text) < minimum_characters:
                    skipped_short += 1
                    continue
                compared_texts[field] += 1
                variants = _search_variants(text)
                location = next(
                    (
                        location
                        for location, section in sections
                        if any(variant in section for variant in variants)
                    ),
                    None,
                )
                if location is None:
                    continue
                record_id = record.get("id")
                matches.append(
                    {
                        "split": split_name,
                        "record_index": index,
                        "record_id": None if record_id is None else str(record_id),
                        "field": field,
                        "location": location,
                        "characters": len(text),
                    }
                )
    fields_without_records = [field for field in fields if not records_with_field[field]]
    return {
        "schema_version": "1.0",
        "evaluator_id": definition.evaluator_id,
        "checked_splits": {name: len(records) for name, records in splits.items()},
        "text_fields": fields,
        "records_with_field": records_with_field,
        "compared_texts": compared_texts,
        "fields_without_records": fields_without_records,
        "minimum_characters": minimum_characters,
        "skipped_short_texts": skipped_short,
        "matches": matches,
        "leaked_records": len({(match["split"], match["record_index"]) for match in matches}),
        "clean": not matches and any(compared_texts.values()),
    }


def describe_leakage(report: Mapping[str, Any]) -> str:
    """Summarize a leakage report as one gate-blocker sentence."""
    matches = list(report.get("matches") or [])
    if not matches:
        fields = [str(field) for field in report.get("text_fields") or []]
        compared = report.get("compared_texts") or {}
        present = report.get("records_with_field") or {}
        uncompared = [field for field in fields if not compared.get(field)]
        unchecked = uncompared or fields or ["the record"]
        names = ", ".join(unchecked)
        if any(present.get(field) for field in unchecked):
            return (
                f"leakage check compared no held-out text for {names}: every value was "
                f"shorter than {report.get('minimum_characters')} characters, so lower "
                "the minimum length or compare fields that carry the record text"
            )
        return (
            f"leakage check compared no held-out records for {names}: "
            "point --text-fields at fields these records carry"
        )
    splits = sorted({str(match["split"]) for match in matches})
    locations = ", ".join(
        f"{match['split']}[{match['record_index']}].{match['field']} in {match['location']}"
        for match in matches[:3]
    )
    return (
        f"evaluator definition reuses text from {report.get('leaked_records') or len(matches)} "
        f"held-out {'/'.join(splits) or 'split'} records: {locations}"
        f"{', ...' if len(matches) > 3 else ''}"
    )
