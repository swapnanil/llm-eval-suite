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
        "extraction", "translation", "general"
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
