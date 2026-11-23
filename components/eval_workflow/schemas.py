"""Typed contracts for portable evaluator definitions and validation reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    """Base model for persistent artifacts that must reject unknown fields."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
    )


class FailureModeDefinition(StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_examples: list[str] = Field(default_factory=list)
    source_annotation_ids: list[str] = Field(default_factory=list)


class BinaryDecision(StrictModel):
    type: Literal["binary"] = "binary"
    pass_definition: str = Field(min_length=1)
    fail_definition: str = Field(min_length=1)


class LifecycleCheckpoints(StrictModel):
    human_review_completed: bool
    obvious_errors_reviewed: bool
    expert_examples_include_pass_and_fail: bool
    note: str = Field(min_length=1)


class ExpertExample(StrictModel):
    input: dict[str, Any]
    critique: str = Field(min_length=1)
    result: Literal["Pass", "Fail"]


class StringSchema(StrictModel):
    type: Literal["string"] = "string"
    min_length: Literal[1] = Field(default=1, alias="minLength")


class BinaryResultSchema(StrictModel):
    type: Literal["string"] = "string"
    enum: tuple[Literal["Pass"], Literal["Fail"]] = ("Pass", "Fail")


class BinaryOutputProperties(StrictModel):
    critique: StringSchema = Field(default_factory=StringSchema)
    result: BinaryResultSchema = Field(default_factory=BinaryResultSchema)


class BinaryJudgeOutputSchema(StrictModel):
    type: Literal["object"] = "object"
    additional_properties: Literal[False] = Field(
        default=False, alias="additionalProperties"
    )
    required: tuple[Literal["critique"], Literal["result"]] = ("critique", "result")
    properties: BinaryOutputProperties = Field(default_factory=BinaryOutputProperties)


class RuleDefinition(StrictModel):
    pattern: str = Field(min_length=1)
    match_means: Literal["Fail"] = "Fail"
    case_sensitive: bool = False


class ConfusionMatrix(StrictModel):
    true_pass: int = Field(ge=0)
    false_fail: int = Field(ge=0)
    true_fail: int = Field(ge=0)
    false_pass: int = Field(ge=0)


class ClassMetrics(StrictModel):
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)


class PerClassMetrics(StrictModel):
    pass_metrics: ClassMetrics = Field(alias="Pass")
    fail_metrics: ClassMetrics = Field(alias="Fail")


class ClassCounts(StrictModel):
    pass_count: int = Field(alias="Pass", ge=0)
    fail_count: int = Field(alias="Fail", ge=0)


class BinaryClassificationMetrics(StrictModel):
    examples: int = Field(ge=1)
    confusion_matrix: ConfusionMatrix
    tpr: float | None = Field(default=None, ge=0, le=1)
    tnr: float | None = Field(default=None, ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    balanced_accuracy: float | None = Field(default=None, ge=0, le=1)
    per_class: PerClassMetrics
    class_counts: ClassCounts


class ConfidenceIntervals(StrictModel):
    tpr: tuple[float, float] | None = None
    tnr: tuple[float, float] | None = None


class ValidationThresholds(StrictModel):
    minimum_tpr: float = Field(ge=0, le=1)
    minimum_tnr: float = Field(ge=0, le=1)
    cli_minimum_tpr: float = Field(ge=0, le=1)
    cli_minimum_tnr: float = Field(ge=0, le=1)
    evaluator_minimum_tpr: float | None = Field(default=None, ge=0, le=1)
    evaluator_minimum_tnr: float | None = Field(default=None, ge=0, le=1)


class SplitProvenance(StrictModel):
    verified: Literal[True]
    split_manifest: str = Field(min_length=1)
    test_artifact: str = Field(min_length=1)
    verified_fields: list[str]
    excluded_prediction_field: str = Field(min_length=1)
    record_count: int = Field(ge=1)


class LeakageMatch(StrictModel):
    split: str = Field(min_length=1)
    record_index: int = Field(ge=0)
    record_id: str | None = None
    field: str = Field(min_length=1)
    location: str = Field(min_length=1)
    characters: int = Field(ge=1)


class LeakageReport(StrictModel):
    schema_version: Literal["1.0"]
    evaluator_id: str = Field(min_length=1)
    checked_splits: dict[str, int]
    text_fields: list[str] = Field(min_length=1)
    records_with_field: dict[str, int]
    compared_texts: dict[str, int]
    fields_without_records: list[str] = Field(default_factory=list)
    minimum_characters: int = Field(ge=1)
    skipped_short_texts: int = Field(ge=0)
    matches: list[LeakageMatch] = Field(default_factory=list)
    leaked_records: int = Field(ge=0)
    clean: bool


class GroupValidationReport(StrictModel):
    dimensions: dict[str, str]
    metrics: BinaryClassificationMetrics


class JudgeValidationReport(StrictModel):
    schema_version: Literal["1.0"]
    created_at: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    dataset_role: Literal["held_out_test"]
    provenance: SplitProvenance
    human_label_column: str = Field(min_length=1)
    evaluator_label_column: str = Field(min_length=1)
    metrics: BinaryClassificationMetrics
    confidence_intervals_95: ConfidenceIntervals
    thresholds: ValidationThresholds
    gate_passed: bool
    groups: list[GroupValidationReport] = Field(default_factory=list)
    gate_blockers: list[str] = Field(default_factory=list)
    labels_sha256: str | None = None
    leakage: LeakageReport | None = None


class SuccessRateEstimate(StrictModel):
    population_size: int = Field(ge=1)
    observed_pass_rate: float = Field(ge=0, le=1)
    corrected_pass_rate: float = Field(ge=0, le=1)
    confidence_interval_95: tuple[float, float]
    judge_tpr: float = Field(ge=0, le=1)
    judge_tnr: float = Field(ge=0, le=1)
    test_examples: int = Field(ge=1)
    bootstrap_iterations: int = Field(ge=1)
    bootstrap_samples_uncorrectable: int = Field(ge=0)


class SuccessRateReport(StrictModel):
    schema_version: Literal["1.0"]
    created_at: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    test_set: str = Field(min_length=1)
    provenance: SplitProvenance
    judge_label_column: str = Field(min_length=1)
    evaluator: str | None = None
    evaluator_id: str | None = None
    dataset_evaluator_column: str | None = None
    estimate: SuccessRateEstimate
    minimum_pass_rate: float | None = Field(default=None, ge=0, le=1)
    gate_passed: bool | None = None
    gate_blockers: list[str] = Field(default_factory=list)


class McNemarResult(StrictModel):
    paired_rows: int = Field(ge=1)
    discordant_a_only: int = Field(ge=0)
    discordant_b_only: int = Field(ge=0)
    p_value: float = Field(ge=0, le=1)


class PairedPowerPlan(StrictModel):
    alpha: float = Field(gt=0, lt=1)
    power: float = Field(gt=0, lt=1)
    item_difference_sd: float = Field(ge=0)
    minimum_detectable_effect: float = Field(ge=0)
    target_effect: float | None = Field(default=None, gt=0)
    required_items: int | None = Field(default=None, ge=1)


class PairedComparison(StrictModel):
    items: int = Field(ge=2)
    paired_rows: int = Field(ge=2)
    runs_per_item_min: int = Field(ge=1)
    runs_per_item_max: int = Field(ge=1)
    excluded_unpaired_items: int = Field(ge=0)
    pass_rate_a: float = Field(ge=0, le=1)
    pass_rate_b: float = Field(ge=0, le=1)
    mean_difference: float = Field(ge=-1, le=1)
    confidence_interval_95: tuple[float, float]
    bootstrap_iterations: int = Field(ge=1)
    seed: int
    clustered_standard_error: float = Field(ge=0)
    naive_standard_error: float = Field(ge=0)
    design_effect: float = Field(ge=0)
    mcnemar: McNemarResult
    power: PairedPowerPlan


class PairedComparisonReport(StrictModel):
    schema_version: Literal["1.0"]
    created_at: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    item_column: str = Field(min_length=1)
    run_column: str = Field(min_length=1)
    outcome_column: str = Field(min_length=1)
    run_a: str = Field(min_length=1)
    run_b: str = Field(min_length=1)
    comparison: PairedComparison


class EvaluatorValidation(StrictModel):
    required: Literal[True] = True
    validated: bool = False
    minimum_tpr: float = Field(default=0.8, ge=0, le=1)
    minimum_tnr: float = Field(default=0.8, ge=0, le=1)
    note: str = Field(min_length=1)
    last_report: JudgeValidationReport | None = None
    history: list[JudgeValidationReport] = Field(default_factory=list)


class EvaluatorDefinition(StrictModel):
    """Versioned evaluator artifact used by the CLI, app, and workspace store."""

    schema_version: Literal["1.0"]
    evaluator_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(min_length=1)
    kind: Literal["llm_judge", "regex", "contains"]
    status: Literal["draft_unvalidated", "validated", "validation_failed"]
    input_fields: list[str] = Field(min_length=1)
    failure_mode: FailureModeDefinition
    decision: BinaryDecision
    validation: EvaluatorValidation
    lifecycle_checkpoints: LifecycleCheckpoints
    expert_examples: list[ExpertExample] = Field(default_factory=list)
    created_at: str = Field(min_length=1)
    prompt: str | None = None
    output_schema: BinaryJudgeOutputSchema | None = None
    rule: RuleDefinition | None = None

    @model_validator(mode="after")
    def kind_specific_contract_is_complete(self) -> EvaluatorDefinition:
        if len(set(self.input_fields)) != len(self.input_fields) or any(
            not field.strip() for field in self.input_fields
        ):
            raise ValueError("Evaluator input_fields must be unique, non-empty strings")
        if self.kind == "llm_judge":
            if self.prompt is None or self.prompt.count("{{evaluation_input}}") != 1:
                raise ValueError(
                    "LLM judge prompt must contain exactly one {{evaluation_input}} placeholder"
                )
            if self.output_schema is None:
                raise ValueError("LLM judge requires the strict binary output schema")
            if self.rule is not None:
                raise ValueError("LLM judge must not define a deterministic rule")
        elif self.rule is None:
            raise ValueError("Rule evaluator definition requires a rule")
        elif self.prompt is not None or self.output_schema is not None:
            raise ValueError(
                "Rule evaluator must not define an LLM prompt or output schema"
            )
        return self


def parse_evaluator_definition(
    value: Mapping[str, Any] | EvaluatorDefinition,
    *,
    required_kind: str | None = None,
) -> EvaluatorDefinition:
    """Parse an evaluator artifact once at a trust boundary."""
    try:
        evaluator = (
            value
            if isinstance(value, EvaluatorDefinition)
            else EvaluatorDefinition.model_validate(value)
        )
    except ValidationError as error:
        raise ValueError(f"Invalid evaluator definition: {error}") from error
    if required_kind is not None and evaluator.kind != required_kind:
        raise ValueError(f"Evaluator kind must be {required_kind}")
    return evaluator


def dump_evaluator_definition(evaluator: EvaluatorDefinition) -> dict[str, Any]:
    """Serialize an evaluator artifact with its public JSON field names."""
    return evaluator.model_dump(mode="json", by_alias=True, exclude_none=True)
