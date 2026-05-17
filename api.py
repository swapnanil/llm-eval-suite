from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.comparator import attach_pairwise_to_report
from agent.evaluator import evaluate
from agent.models import BatchEvalSummary, EvalInput, EvalReport, LLMResponse
from agent.prompts import TASK_DIMENSIONS, DIMENSION_DESCRIPTIONS

app = FastAPI(
    title="LLM Output Eval Suite",
    description="Evaluate and compare LLM outputs with structured, multi-dimensional scoring.",
    version="1.0.0",
)

logger = logging.getLogger(__name__)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": os.environ.get("MODEL", "claude-sonnet-4-6"),
    }


@app.get("/dimensions")
def get_dimensions() -> dict:
    result = {}
    for task_type, dims in TASK_DIMENSIONS.items():
        result[task_type] = {
            dim: {
                "weight": weight,
                "description": DIMENSION_DESCRIPTIONS.get(dim, f"Evaluate {dim}."),
            }
            for dim, weight in dims.items()
        }
    return result


@app.post("/eval", response_model=EvalReport)
def eval_endpoint(eval_input: EvalInput) -> EvalReport:
    try:
        report = evaluate(eval_input)
        if eval_input.eval_mode in ("compare", "rank") and len(eval_input.responses) > 1:
            report = attach_pairwise_to_report(report, eval_input)
        return report
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


class QuickEvalRequest(BaseModel):
    prompt: str
    responses: list[LLMResponse]
    task_type: str = "general"


@app.post("/eval/quick", response_model=EvalReport)
def eval_quick(req: QuickEvalRequest) -> EvalReport:
    try:
        eval_input = EvalInput(
            prompt=req.prompt,
            task_type=req.task_type,
            responses=req.responses,
            eval_mode="score" if len(req.responses) == 1 else "rank",
        )
        report = evaluate(eval_input)
        if eval_input.eval_mode in ("compare", "rank") and len(eval_input.responses) > 1:
            report = attach_pairwise_to_report(report, eval_input)
        return report
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval/batch", response_model=BatchEvalSummary)
def eval_batch(inputs: list[EvalInput]) -> BatchEvalSummary:
    if not inputs:
        raise HTTPException(status_code=422, detail="At least one eval input is required")

    reports: list[EvalReport] = []
    for eval_input in inputs:
        try:
            report = evaluate(eval_input)
            if eval_input.eval_mode in ("compare", "rank") and len(eval_input.responses) > 1:
                report = attach_pairwise_to_report(report, eval_input)
            reports.append(report)
        except Exception as e:
            logger.error("Batch eval failed for one item: %s", e)
            raise HTTPException(status_code=500, detail=f"Batch eval failed: {e}")

    insights: list[str] = []
    if reports:
        winners = [r.winner for r in reports if r.winner]
        if winners:
            from collections import Counter
            top = Counter(winners).most_common(1)[0]
            insights.append(f"{top[0]} won {top[1]}/{len(reports)} evaluations.")
        insights.append(f"Evaluated {len(reports)} prompts across {len(set(r.task_type for r in reports))} task type(s).")

    return BatchEvalSummary(
        total_evals=len(reports),
        reports=reports,
        aggregate_insights=insights,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
