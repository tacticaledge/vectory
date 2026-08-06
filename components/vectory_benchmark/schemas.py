"""Pydantic schemas for VectoryBenchmark tasks, traces, and scoring."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentDomain(str, Enum):
    """Top-level task domains covered by VectoryBenchmark."""

    EXECUTION = "execution"
    RESEARCH = "research"
    CODING = "coding"
    DATA_REASONING = "data_reasoning"
    WORKFLOW = "workflow"
    SAFETY_BOUNDARY = "safety_boundary"
    LONG_CONTEXT = "long_context"
    RECOVERY = "recovery"
    FRONTEND = "frontend"


class EventType(str, Enum):
    """Normalized trace event types."""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE_EDIT = "file_edit"
    TEST_RUN = "test_run"
    VERIFICATION = "verification"
    DECISION = "decision"
    CLAIM = "claim"
    EVIDENCE = "evidence"
    PROOF_OBLIGATION = "proof_obligation"
    CHECKER_RESULT = "checker_result"
    POLICY_CHECK = "policy_check"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    FINAL = "final"


class Severity(str, Enum):
    """Pathology severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScoreBand(str, Enum):
    """Human-readable score bands."""

    EXCELLENT = "excellent"
    STRONG = "strong"
    MIXED = "mixed"
    WEAK = "weak"
    FAILING = "failing"


class ScoreWeights(BaseModel):
    """Weights for the Vectory Score dimensions."""

    model_config = ConfigDict(extra="forbid")

    task_success: float = 0.23
    reality_sampling: float = 0.14
    trace_productivity: float = 0.14
    tool_retrieval_discipline: float = 0.13
    recovery: float = 0.09
    agent_control: float = 0.10
    turn_efficiency: float = 0.05
    evidence_quality: float = 0.05
    proof_grounding: float = 0.07

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ScoreWeights":
        total = (
            self.task_success
            + self.reality_sampling
            + self.trace_productivity
            + self.tool_retrieval_discipline
            + self.recovery
            + self.agent_control
            + self.turn_efficiency
            + self.evidence_quality
            + self.proof_grounding
        )
        if abs(total - 1.0) > 0.0001:
            raise ValueError("Score weights must sum to 1.0")
        return self


class FormalCheckerConfig(BaseModel):
    """Trusted suite-defined checker command for optional local formal execution."""

    model_config = ConfigDict(extra="forbid")

    name: str
    checker_type: str = "generic"
    command: list[str]
    obligation_ids: list[str] = Field(default_factory=list)
    timeout_seconds: int = 60

    @field_validator("command")
    @classmethod
    def command_is_explicit_argv(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("formal checker command must not be empty")
        if any(not str(part).strip() for part in value):
            raise ValueError("formal checker command entries must be non-empty")
        return value


class TaskLimits(BaseModel):
    """Operational limits expected for a task."""

    model_config = ConfigDict(extra="forbid")

    max_events: int = 80
    max_tool_calls: int = 40
    turn_cliff_events: int = 40
    max_wall_time_seconds: int | None = None
    max_tokens: int | None = None


class TaskChecks(BaseModel):
    """Deterministic checks used by the local scorer."""

    model_config = ConfigDict(extra="forbid")

    expected_answer_keywords: list[str] = Field(default_factory=list)
    forbidden_answer_patterns: list[str] = Field(default_factory=list)
    required_event_types: list[EventType] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    reality_sampling_tools: list[str] = Field(default_factory=list)
    retrieval_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    evidence_markers: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=list)
    forbidden_paths: list[str] = Field(default_factory=list)
    min_reality_events: int = 1
    min_evidence_events: int = 0
    requires_verification: bool = False
    requires_recovery: bool = False
    requires_approval: bool = False
    requires_retrieval: bool = False
    requires_proof: bool = False
    min_proof_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    required_obligation_ids: list[str] = Field(default_factory=list)
    accepted_checker_types: list[str] = Field(default_factory=list)
    allow_placeholder_proofs: bool = False
    formal_checkers: list[FormalCheckerConfig] = Field(default_factory=list)
    penalize_premature_planning: bool = True


class BenchmarkTask(BaseModel):
    """A single VectoryBenchmark task definition."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    domain: AgentDomain
    difficulty: Literal["calibration", "standard", "advanced", "stress"] = "standard"
    intent: str
    setup: str
    success_criteria: list[str]
    adversarial_pressures: list[str] = Field(default_factory=list)
    checks: TaskChecks = Field(default_factory=TaskChecks)
    limits: TaskLimits = Field(default_factory=TaskLimits)
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    tags: list[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def task_id_is_stable(cls, value: str) -> str:
        if not value or value.strip() != value or " " in value:
            raise ValueError("task_id must be a stable non-empty identifier without spaces")
        return value


class BenchmarkSuite(BaseModel):
    """A versioned collection of benchmark tasks."""

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    version: str
    title: str
    description: str
    tasks: list[BenchmarkTask]

    @model_validator(mode="after")
    def task_ids_are_unique(self) -> "BenchmarkSuite":
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("BenchmarkSuite task IDs must be unique")
        return self


class TraceEvent(BaseModel):
    """A normalized event from an agent trajectory."""

    model_config = ConfigDict(extra="allow")

    type: EventType
    content: str = ""
    name: str | None = None
    input: Any = None
    output: Any = None
    path: str | None = None
    success: bool | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    """A claim made by the agent that should be backed by evidence or proof."""

    model_config = ConfigDict(extra="allow")

    claim_id: str
    text: str
    source_event_id: str | None = None
    final_answer_span: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    obligation_ids: list[str] = Field(default_factory=list)


class EvidenceReference(BaseModel):
    """Evidence used to support one or more agent claims."""

    model_config = ConfigDict(extra="allow")

    evidence_id: str
    source_type: str = "trace"
    event_id: str | None = None
    path: str | None = None
    output_span: str | None = None
    artifact_path: str | None = None
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    supports_claim_ids: list[str] = Field(default_factory=list)


class ProofObligation(BaseModel):
    """A proof obligation or invariant that must be closed for proof-sensitive tasks."""

    model_config = ConfigDict(extra="allow")

    obligation_id: str
    description: str
    checker_type: str = "generic"
    status: Literal["open", "closed", "failed", "waived", "unknown"] = "unknown"
    depends_on: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class CheckerResult(BaseModel):
    """A checker result supplied by a trace or produced by a trusted local checker."""

    model_config = ConfigDict(extra="allow")

    checker_id: str
    name: str
    checker_type: str = "generic"
    status: Literal["passed", "failed", "error", "skipped", "unknown"] = "unknown"
    obligation_ids: list[str] = Field(default_factory=list)
    output_summary: str = ""
    artifact_path: str | None = None
    command: list[str] | None = None


class AgentRun(BaseModel):
    """A submitted agent run for one benchmark task."""

    model_config = ConfigDict(extra="allow")

    agent: str
    model: str
    task_id: str
    run_id: str
    final_answer: str = ""
    status: Literal["completed", "failed", "timeout", "aborted", "unknown"] = "unknown"
    events: list[TraceEvent] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    proof_obligations: list[ProofObligation] = Field(default_factory=list)
    checker_results: list[CheckerResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    declared_success: bool | None = None

    @field_validator("events", mode="before")
    @classmethod
    def normalize_events(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("events must be a list")
        return value


class PathologyFinding(BaseModel):
    """A detected agent behavior pathology."""

    model_config = ConfigDict(extra="forbid")

    code: str
    name: str
    severity: Severity
    score_penalty: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    recommendation: str = ""


class DimensionScore(BaseModel):
    """A single score dimension."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    rationale: str


class RunScore(BaseModel):
    """Score report for one agent run."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    model: str
    task_id: str
    run_id: str
    domain: AgentDomain
    vectory_score: float = Field(ge=0.0, le=1.0)
    band: ScoreBand
    passed: bool
    dimensions: dict[str, DimensionScore]
    pathologies: list[PathologyFinding] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


class LeaderboardEntry(BaseModel):
    """Aggregated leaderboard row."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    agent: str
    model: str
    runs: int
    tasks: int
    vectory_score: float
    pass_at_1: float
    robust_pass_at_5: float
    productive_work_ratio: float
    pathology_risk: float
    agent_control_index: float
