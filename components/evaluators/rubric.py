"""Checklist-decomposed rubric judging: per-criterion verdicts and a weighted rollup.

A rubric judge decides an explicit list of atomic yes/no criteria instead of one
holistic score. The rollup is the clipped weighted fraction of the criteria the judge
resolved, so the diagnostics stay per criterion and the score stays reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# Per-criterion outcomes. "unknown" covers a criterion the judge skipped, repeated, or
# returned unparseably: it never counts as met and never reaches the score denominator.
MET = "met"
NOT_MET = "not_met"
UNKNOWN = "unknown"

_MET_LABELS = {"met", "yes", "true", "pass", "1"}
_NOT_MET_LABELS = {"not_met", "not met", "no", "false", "fail", "0"}


class StrictModel(BaseModel):
    """Base model for rubric artifacts that must reject unknown fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, validate_default=True)


class RubricCriterion(StrictModel):
    """One atomic yes/no checklist item and its contribution to the score."""

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    text: str = Field(min_length=1)
    weight: float = Field(ge=0)
    polarity: Literal["positive", "negative"] = "positive"

    @field_validator("weight", mode="before")
    @classmethod
    def weight_is_a_finite_number(cls, value: Any) -> Any:
        # Strings and booleans would otherwise be coerced into a silently different weight.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("criterion weight must be a number")
        if not math.isfinite(float(value)):
            raise ValueError("criterion weight must be finite")
        return value


class RubricSpec(StrictModel):
    """Validated checklist a rubric judge scores one response against."""

    schema_version: Literal["1.0"] = "1.0"
    rubric_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    criteria: list[RubricCriterion] = Field(min_length=1)
    description: str | None = None

    @model_validator(mode="after")
    def criteria_can_produce_a_score(self) -> RubricSpec:
        counts = Counter(criterion.id for criterion in self.criteria)
        repeated = sorted(identifier for identifier, count in counts.items() if count > 1)
        if repeated:
            raise ValueError(
                f"Rubric criterion ids must be unique; repeated: {', '.join(repeated)}"
            )
        if self.positive_weight() <= 0:
            raise ValueError(
                "Rubric needs at least one positive criterion with a weight above zero; "
                "an all-zero weight set leaves the score denominator empty"
            )
        return self

    def positive_weight(self) -> float:
        """Return the total weight a fully met response could earn."""
        return sum(
            criterion.weight for criterion in self.criteria if criterion.polarity == "positive"
        )

    def fingerprint(self) -> str:
        """Identify the complete definition independently of its human-readable ID."""
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()


class RubricCriterionVerdict(StrictModel):
    id: str = Field(min_length=1)
    met: bool


class RubricJudgeResponse(StrictModel):
    """Requested response contract for a checklist judge.

    Only the provider-side contract is strict. Parsing stays lenient so one skipped or
    malformed item becomes "unknown" instead of discarding the whole checklist.
    """

    critique: str = Field(min_length=1)
    criteria: list[RubricCriterionVerdict] = Field(min_length=1)


class RubricCriterionReport(StrictModel):
    id: str
    text: str
    weight: float
    polarity: Literal["positive", "negative"]
    evaluated: int = Field(ge=0)
    met: int = Field(ge=0)
    not_met: int = Field(ge=0)
    unknown: int = Field(ge=0)
    failures: int = Field(ge=0)
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    unknown_rate: float | None = Field(default=None, ge=0, le=1)


class RubricReport(StrictModel):
    schema_version: Literal["1.0"]
    rubric_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    rubric_id: str
    responses: int = Field(ge=0)
    scored_responses: int = Field(ge=0)
    unscored_responses: int = Field(ge=0)
    unparseable_responses: int = Field(ge=0)
    mean_score: float | None = Field(default=None, ge=0, le=1)
    criteria: list[RubricCriterionReport] = Field(min_length=1)


def parse_rubric_spec(value: Mapping[str, Any] | RubricSpec) -> RubricSpec:
    """Validate a rubric spec once at a trust boundary."""
    try:
        return value if isinstance(value, RubricSpec) else RubricSpec.model_validate(value)
    except ValidationError as error:
        raise ValueError(f"Invalid rubric spec: {error}") from error


def render_rubric_checklist(spec: Mapping[str, Any] | RubricSpec) -> str:
    """Render the checklist section of a rubric judge prompt."""
    rubric = parse_rubric_spec(spec)
    lines = []
    for criterion in rubric.criteria:
        penalty = (
            " [counts against the response when met]"
            if criterion.polarity == "negative"
            else ""
        )
        lines.append(f"- {criterion.id} (weight {criterion.weight:g}){penalty}: {criterion.text}")
    return "\n".join(lines)


def _normalize_met(value: Any) -> bool | None:
    """Return the met/not-met verdict a judge expressed, or None when it is unusable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _MET_LABELS:
            return True
        if normalized in _NOT_MET_LABELS:
            return False
    return None


def _unusable_response(critique: str | None, reason: str | None) -> dict[str, Any]:
    return {
        "critique": critique,
        "verdicts": {},
        "repeated_ids": [],
        "unparsed_entries": 0,
        "parse_error": reason,
    }


def _response_verdicts(response_text: str) -> dict[str, Any]:
    """Collect the usable per-criterion verdicts from one recorded judge response.

    The contract is one verdict per listed id, so a repeated item is reported instead of
    resolved whether or not its verdicts agree: agreeing repeats are still a response
    that did not decide the checklist as asked, and resolving them would keep that out
    of the diagnostics while contradictory repeats are reported.
    """
    try:
        payload = json.loads(response_text)
    except (TypeError, ValueError):
        return _unusable_response(None, "Rubric judge response is not JSON")
    if not isinstance(payload, Mapping):
        return _unusable_response(None, "Rubric judge response is not a JSON object")
    critique = payload.get("critique")
    critique = critique.strip() if isinstance(critique, str) and critique.strip() else None
    if critique is None:
        # The critique is the diagnostic a checklist judge exists for. A verdict list
        # without one is not a usable judgment, so it stays unscored and out of the
        # reuse cache instead of being recorded as a score with no reasoning.
        return _unusable_response(None, "Rubric judge response has no critique")
    entries = payload.get("criteria")
    if not isinstance(entries, list):
        return _unusable_response(critique, "Rubric judge response has no criteria list")

    observed: dict[str, list[bool]] = {}
    unparsed_entries = 0
    for entry in entries:
        identifier = entry.get("id") if isinstance(entry, Mapping) else None
        met = _normalize_met(entry.get("met")) if isinstance(entry, Mapping) else None
        if not isinstance(identifier, str) or not identifier.strip() or met is None:
            unparsed_entries += 1
            continue
        observed.setdefault(identifier.strip(), []).append(met)
    return {
        "critique": critique,
        "verdicts": {
            identifier: verdicts[0]
            for identifier, verdicts in observed.items()
            if len(verdicts) == 1
        },
        "repeated_ids": sorted(
            identifier for identifier, verdicts in observed.items() if len(verdicts) > 1
        ),
        "unparsed_entries": unparsed_entries,
        "parse_error": None,
    }


def _weighted_score(
    rubric: RubricSpec, statuses: Mapping[str, str]
) -> tuple[float, float, float | None]:
    """Return the met weight, the denominator weight, and the clipped rollup.

    Sharing one implementation keeps a scored response and the report derived from it in
    agreement: the parser and the aggregator both read the score off the same verdicts.
    """
    weight_met = 0.0
    weight_possible = 0.0
    for criterion in rubric.criteria:
        status = statuses[criterion.id]
        if criterion.polarity == "positive" and status != UNKNOWN:
            weight_possible += criterion.weight
        if status == MET:
            weight_met += (
                criterion.weight if criterion.polarity == "positive" else -criterion.weight
            )
    score = min(max(weight_met / weight_possible, 0.0), 1.0) if weight_possible > 0 else None
    return weight_met, weight_possible, score


def _recorded_score(value: Any, index: int) -> float | None:
    """Return the score a result recorded, rejecting a value that is not a number."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Result {index} records a non-numeric score: {value!r}")
    return float(value)


def parse_rubric_response(
    response_text: str, spec: Mapping[str, Any] | RubricSpec
) -> dict[str, Any]:
    """Score one recorded judge response against a rubric checklist.

    The score is the clipped weighted fraction ``sum(w * met) / sum(w)`` over the
    positive criteria the judge resolved; a met negative criterion subtracts its weight
    from the numerator. Criteria the judge skipped, repeated, or returned unparseably are
    reported as "unknown" and stay out of both sums, so an incomplete response reduces
    coverage instead of silently passing or failing its items.
    """
    rubric = parse_rubric_spec(spec)
    parsed = _response_verdicts(response_text)
    verdicts = parsed["verdicts"]

    criteria = []
    counts = {MET: 0, NOT_MET: 0, UNKNOWN: 0}
    statuses: dict[str, str] = {}
    for criterion in rubric.criteria:
        met = verdicts.get(criterion.id)
        status = UNKNOWN if met is None else (MET if met else NOT_MET)
        counts[status] += 1
        statuses[criterion.id] = status
        criteria.append(
            {
                "id": criterion.id,
                "status": status,
                "weight": criterion.weight,
                "polarity": criterion.polarity,
            }
        )
    weight_met, weight_possible, score = _weighted_score(rubric, statuses)
    declared = {criterion.id for criterion in rubric.criteria}
    return {
        "rubric_id": rubric.rubric_id,
        "rubric_fingerprint": rubric.fingerprint(),
        "score": score,
        "critique": parsed["critique"],
        "weight_met": weight_met,
        "weight_possible": weight_possible,
        "criteria": criteria,
        "status_counts": counts,
        "repeated_ids": [
            identifier for identifier in parsed["repeated_ids"] if identifier in declared
        ],
        "unexpected_ids": sorted((set(verdicts) | set(parsed["repeated_ids"])) - declared),
        "unparsed_entries": parsed["unparsed_entries"],
        "parse_error": parsed["parse_error"],
    }


def aggregate_rubric_results(
    spec: Mapping[str, Any] | RubricSpec, results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Rank rubric criteria by their dataset-level failure rate.

    A positive criterion fails when it was not met; a negative criterion fails when the
    judge found it met, because a met penalty is the defect it names. Rates are measured
    over the responses that resolved the criterion, so unknown items lower coverage
    instead of the rate. The report is derived only from the supplied results and carries
    no timestamps, so two runs over the same recorded outputs are byte-identical.

    Every result must record the rubric it was scored against, because two rubrics can
    share criterion ids while weighting or inverting them differently, and must report
    every criterion exactly once, because a repeated or omitted criterion would count a
    single response more than once or leave it out of the coverage it reports. The mean
    is recomputed from the reported verdicts rather than taken on trust, and a result
    whose recorded score contradicts them is rejected, so the reported mean can never
    disagree with the criterion statistics beside it.
    """
    rubric = parse_rubric_spec(spec)
    fingerprint = rubric.fingerprint()
    tallies = {criterion.id: {MET: 0, NOT_MET: 0, UNKNOWN: 0} for criterion in rubric.criteria}
    rows = list(results)
    scores = []
    unparseable = 0
    for index, result in enumerate(rows):
        scored_against = result.get("rubric_id")
        if scored_against != rubric.rubric_id:
            raise ValueError(
                f"Result {index} was scored against rubric {scored_against!r}, "
                f"not {rubric.rubric_id!r}; shared criterion ids do not make two "
                "rubrics comparable"
            )
        if result.get("rubric_fingerprint") != fingerprint:
            raise ValueError(f"Result {index} has a different rubric fingerprint; re-evaluate it")
        if result.get("parse_error"):
            unparseable += 1
        statuses: dict[str, str] = {}
        reported: set[str] = set()
        for item in result.get("criteria") or []:
            identifier = str(item.get("id"))
            tally = tallies.get(identifier)
            if tally is None:
                raise ValueError(
                    f"Result criterion {item.get('id')!r} is not part of rubric {rubric.rubric_id}"
                )
            if identifier in reported:
                raise ValueError(
                    f"Result {index} reports criterion {identifier!r} more than once; "
                    "a repeated criterion would count one response several times"
                )
            reported.add(identifier)
            status = str(item.get("status"))
            if status not in tally:
                raise ValueError(f"Unsupported criterion status: {item.get('status')!r}")
            tally[status] += 1
            statuses[identifier] = status
        omitted = [criterion.id for criterion in rubric.criteria if criterion.id not in reported]
        if omitted:
            raise ValueError(
                f"Result {index} omits rubric criteria: {', '.join(omitted)}; "
                "every criterion must be reported, as unknown, so coverage stays accurate"
            )
        score = _recorded_score(result.get("score"), index)
        derived = _weighted_score(rubric, statuses)[2]
        contradicts = (score is None) != (derived is None) or (
            score is not None
            and derived is not None
            and not math.isclose(score, derived, rel_tol=1e-9, abs_tol=1e-12)
        )
        if contradicts:
            raise ValueError(
                f"Result {index} records score {result.get('score')!r}, but its criterion "
                f"verdicts weigh out to {derived!r} under rubric {rubric.rubric_id}; "
                "the mean must not contradict the verdicts it summarizes"
            )
        if derived is not None:
            scores.append(derived)

    criteria = []
    for criterion in rubric.criteria:
        tally = tallies[criterion.id]
        evaluated = tally[MET] + tally[NOT_MET]
        observed = evaluated + tally[UNKNOWN]
        failures = tally[MET] if criterion.polarity == "negative" else tally[NOT_MET]
        criteria.append(
            {
                "id": criterion.id,
                "text": criterion.text,
                "weight": criterion.weight,
                "polarity": criterion.polarity,
                "evaluated": evaluated,
                "met": tally[MET],
                "not_met": tally[NOT_MET],
                "unknown": tally[UNKNOWN],
                "failures": failures,
                "failure_rate": failures / evaluated if evaluated else None,
                "unknown_rate": tally[UNKNOWN] / observed if observed else None,
            }
        )
    # Heaviest first among equal rates; a criterion no response resolved has no rate and ranks last.
    criteria.sort(
        key=lambda item: (
            item["failure_rate"] is None,
            -(item["failure_rate"] or 0.0),
            -item["weight"],
            item["id"],
        )
    )
    return {
        "schema_version": "1.0",
        "rubric_id": rubric.rubric_id,
        "rubric_fingerprint": fingerprint,
        "responses": len(rows),
        "scored_responses": len(scores),
        "unscored_responses": len(rows) - len(scores),
        "unparseable_responses": unparseable,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "criteria": criteria,
    }
