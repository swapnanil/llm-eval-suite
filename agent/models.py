from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, field_validator, model_validator


class LLMResponse(BaseModel):
    label: str
    response: str
    model: str | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None


class EvalInput(BaseModel):
    prompt: str
    task_type: Literal[
        "summarisation", "qa", "instruction_following",
        "creative_writing", "code_generation", "classification",
        "extraction", "translation", "rag", "general"
    ]
    reference_answer: str | None = None
    source_document: str | None = None

    responses: list[LLMResponse]

    dimensions: list[str] | None = None
    weights: dict[str, float] | None = None
    eval_mode: Literal["score", "compare", "rank"] = "score"

    audience: str | None = None
    success_criteria: str | None = None

    @field_validator("responses")
    @classmethod
    def validate_response_count(cls, v: list[LLMResponse]) -> list[LLMResponse]:
        if len(v) == 0:
            raise ValueError("At least one response is required")
        if len(v) > 10:
            raise ValueError("Maximum 10 responses allowed per evaluation")
        return v

    @model_validator(mode="after")
    def validate_response_text(self) -> EvalInput:
        for r in self.responses:
            if not r.response or not r.response.strip():
                raise ValueError(f"Response '{r.label}' has empty text")
        return self


class DimensionEval(BaseModel):
    dimension: str
    score: float
    reasoning: str
    quote: str | None = None


class ResponseEval(BaseModel):
    label: str
    overall_score: float
    dimensions: list[DimensionEval]
    strengths: list[str]
    weaknesses: list[str]
    one_line_verdict: str


class PairwiseResult(BaseModel):
    response_a: str
    response_b: str
    winner: str
    margin: Literal["clear", "slight", "tie"]
    reasoning: str


class EvalReport(BaseModel):
    task_type: str
    eval_mode: str
    timestamp: str

    evaluations: list[ResponseEval]

    ranking: list[str] | None = None
    winner: str | None = None
    winner_reasoning: str | None = None
    pairwise: list[PairwiseResult] | None = None

    key_insight: str
    recommendation: str

    efficiency_note: str | None = None


class BatchEvalSummary(BaseModel):
    total_evals: int
    reports: list[EvalReport]
    aggregate_insights: list[str]


# ── Feature 1: Regression Testing ─────────────────────────────────────────────

class BaselineRecord(BaseModel):
    baseline_id: str
    description: str
    created_at: str
    report: EvalReport


class RegressionCheck(BaseModel):
    dimension: str
    baseline_score: float
    new_score: float
    delta: float
    passed: bool
    threshold_used: float


class RegressionResult(BaseModel):
    baseline_id: str
    passed_overall: bool
    checks: list[RegressionCheck]
    regressions: list[str]
    improvements: list[str]
    summary: str


# ── Feature 2: Hallucination Detection ────────────────────────────────────────

class SuspiciousClaim(BaseModel):
    claim: str
    severity: Literal["critical", "moderate", "low"]
    is_supported: bool
    evidence_or_gap: str


class HallucinationReport(BaseModel):
    response_label: str
    risk_level: Literal["none", "low", "moderate", "high", "critical"]
    hallucination_rate: float
    safe_to_use: bool
    suspicious_claims: list[SuspiciousClaim]
    summary: str


# ── Feature 3: Prompt Sensitivity Analysis ────────────────────────────────────

class PromptVariant(BaseModel):
    label: str
    prompt: str
    notes: str | None = None


class PromptScore(BaseModel):
    label: str
    overall_score: float
    dimension_scores: dict[str, float]


class DimensionSensitivity(BaseModel):
    dimension: str
    mean_score: float
    variance: float
    sensitivity: Literal["stable", "moderate", "sensitive"]


class SensitivityReport(BaseModel):
    recommended_prompt: str
    prompt_scores: list[PromptScore]
    dimension_sensitivities: list[DimensionSensitivity]
    most_sensitive_dimensions: list[str]
    stability_summary: str


# ── Feature 5: Multi-Judge Panel ──────────────────────────────────────────────

class JudgeScore(BaseModel):
    judge_id: int
    label: str
    overall_score: float
    dimension_scores: dict[str, float]


class DimensionConsensus(BaseModel):
    dimension: str
    mean_score: float
    variance: float
    disagreement_flag: bool


class ResponseConsensus(BaseModel):
    label: str
    consensus_score: float
    dimension_consensus: list[DimensionConsensus]
    needs_human_review: bool
    reviewer_note: str | None = None


class PanelEvalReport(BaseModel):
    num_judges: int
    eval_mode: str
    task_type: str
    timestamp: str
    response_consensus: list[ResponseConsensus]
    winner: str | None = None
    flagged_for_review: list[str]
    panel_key_insight: str
