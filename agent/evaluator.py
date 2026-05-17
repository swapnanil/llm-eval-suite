from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import anthropic

from .models import (
    DimensionEval,
    EvalInput,
    EvalReport,
    ResponseEval,
)
from .prompts import (
    JUDGE_SYSTEM_PROMPT,
    build_ranking_prompt,
    build_scoring_prompt,
    get_dimensions_for_task,
)

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def _get_model() -> str:
    return os.environ.get("MODEL", "claude-sonnet-4-6")


def _get_max_tokens() -> int:
    return int(os.environ.get("MAX_TOKENS", "3000"))


def _call_judge(user_prompt: str, max_retries: int = 3) -> dict:
    client = _get_client()
    model = _get_model()
    max_tokens = _get_max_tokens()

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("Judge returned invalid JSON on attempt %d, retrying with schema correction", attempt + 1)
            if attempt < max_retries - 1:
                user_prompt = user_prompt + "\n\nIMPORTANT: Your previous response was not valid JSON. Respond ONLY with valid JSON — no prose, no markdown fences."
                time.sleep(0.5)

        except anthropic.RateLimitError as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("Rate limit hit, backing off %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
            time.sleep(wait)

        except anthropic.APIError as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("API error on attempt %d: %s — backing off %ds", attempt + 1, e, wait)
            time.sleep(wait)

    raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_error}")


def _clamp_score(score: float, label: str, dimension: str) -> float:
    if score < 1.0 or score > 10.0:
        logger.warning("Score %.2f out of range for '%s'/'%s', clamping to [1, 10]", score, label, dimension)
        return max(1.0, min(10.0, score))
    return score


def _compute_weighted_score(dimensions: list[DimensionEval], weights: dict[str, float]) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for dim in dimensions:
        w = weights.get(dim.dimension, 0.0)
        if w > 0:
            weighted_sum += dim.score * w
            total_weight += w
    if total_weight == 0:
        scores = [d.score for d in dimensions]
        return sum(scores) / len(scores) if scores else 5.0
    return weighted_sum / total_weight


def _score_single_response(
    eval_input: EvalInput,
    response_idx: int,
    dimensions: dict[str, float],
) -> ResponseEval:
    resp = eval_input.responses[response_idx]

    prompt_text = build_scoring_prompt(
        prompt=eval_input.prompt,
        response_label=resp.label,
        response_text=resp.response,
        dimensions=dimensions,
        task_type=eval_input.task_type,
        reference_answer=eval_input.reference_answer,
        source_document=eval_input.source_document,
        audience=eval_input.audience,
        success_criteria=eval_input.success_criteria,
        model_name=resp.model,
        latency_ms=resp.latency_ms,
        cost_usd=resp.cost_usd,
    )

    raw = _call_judge(prompt_text)

    dim_evals = []
    for d in raw.get("dimensions", []):
        score = _clamp_score(float(d["score"]), resp.label, d["dimension"])
        dim_evals.append(DimensionEval(
            dimension=d["dimension"],
            score=score,
            reasoning=d["reasoning"],
            quote=d.get("quote"),
        ))

    overall = _clamp_score(float(raw.get("overall_score", 5.0)), resp.label, "overall")

    return ResponseEval(
        label=resp.label,
        overall_score=overall,
        dimensions=dim_evals,
        strengths=raw.get("strengths", []),
        weaknesses=raw.get("weaknesses", []),
        one_line_verdict=raw.get("one_line_verdict", ""),
    )


def _build_efficiency_note(eval_input: EvalInput, evaluations: list[ResponseEval]) -> str | None:
    scored_responses = [
        r for r in eval_input.responses
        if r.latency_ms is not None or r.cost_usd is not None
    ]
    if not scored_responses:
        return None

    eval_map = {e.label: e.overall_score for e in evaluations}
    lines = []
    for r in scored_responses:
        score = eval_map.get(r.label, 0.0)
        parts = [f"{r.label}: score={score:.1f}"]
        if r.latency_ms is not None:
            parts.append(f"latency={r.latency_ms}ms")
        if r.cost_usd is not None:
            parts.append(f"cost=${r.cost_usd:.4f}")
        lines.append(", ".join(parts))
    return " | ".join(lines)


def evaluate(eval_input: EvalInput) -> EvalReport:
    if eval_input.dimensions:
        custom_dims = eval_input.dimensions
        default_dims = get_dimensions_for_task(eval_input.task_type)

        if eval_input.weights:
            dimensions = {d: eval_input.weights.get(d, default_dims.get(d, 0.1)) for d in custom_dims}
        else:
            total = len(custom_dims)
            dimensions = {d: 1.0 / total for d in custom_dims}
    else:
        dimensions = get_dimensions_for_task(eval_input.task_type)
        if eval_input.weights:
            dimensions = {k: eval_input.weights.get(k, v) for k, v in dimensions.items()}

    faithfulness_dims = {"faithfulness", "accuracy"}
    needs_source = any(d in faithfulness_dims for d in dimensions)
    if needs_source and not eval_input.reference_answer and not eval_input.source_document:
        logger.warning(
            "Task type '%s' includes faithfulness/accuracy dimensions but no reference_answer or source_document provided. "
            "These dimensions will be scored on internal consistency only.",
            eval_input.task_type,
        )

    evaluations: list[ResponseEval] = []
    for i in range(len(eval_input.responses)):
        logger.info("Scoring response %d/%d: %s", i + 1, len(eval_input.responses), eval_input.responses[i].label)
        ev = _score_single_response(eval_input, i, dimensions)
        evaluations.append(ev)

    ranking: list[str] | None = None
    winner: str | None = None
    winner_reasoning: str | None = None
    key_insight: str
    recommendation: str

    if eval_input.eval_mode == "score":
        sorted_evals = sorted(evaluations, key=lambda e: e.overall_score, reverse=True)
        key_insight = f"{sorted_evals[0].label} achieved the highest overall score of {sorted_evals[0].overall_score:.1f}/10."
        recommendation = (
            f"Use {sorted_evals[0].label}. "
            + (sorted_evals[0].one_line_verdict if sorted_evals[0].one_line_verdict else "")
        )
    elif eval_input.eval_mode in ("compare", "rank"):
        rank_prompt = build_ranking_prompt(
            prompt=eval_input.prompt,
            responses=[{"label": r.label, "response": r.response} for r in eval_input.responses],
            task_type=eval_input.task_type,
            reference_answer=eval_input.reference_answer,
            source_document=eval_input.source_document,
        )
        rank_raw = _call_judge(rank_prompt)
        ranking = rank_raw.get("ranking")
        winner = rank_raw.get("winner")
        winner_reasoning = rank_raw.get("winner_reasoning")
        key_insight = rank_raw.get("key_insight", "")
        recommendation = rank_raw.get("recommendation", "")
    else:
        key_insight = ""
        recommendation = ""

    efficiency_note = _build_efficiency_note(eval_input, evaluations)

    return EvalReport(
        task_type=eval_input.task_type,
        eval_mode=eval_input.eval_mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        evaluations=evaluations,
        ranking=ranking,
        winner=winner,
        winner_reasoning=winner_reasoning,
        pairwise=None,
        key_insight=key_insight,
        recommendation=recommendation,
        efficiency_note=efficiency_note,
    )
