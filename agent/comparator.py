from __future__ import annotations

import itertools
import logging
from collections import defaultdict

from .evaluator import _call_judge
from .models import EvalInput, EvalReport, PairwiseResult, ResponseEval
from .prompts import build_comparison_prompt

logger = logging.getLogger(__name__)


def _pairwise(
    eval_input: EvalInput,
    label_a: str,
    label_b: str,
) -> PairwiseResult:
    resp_a = next(r for r in eval_input.responses if r.label == label_a)
    resp_b = next(r for r in eval_input.responses if r.label == label_b)

    prompt_text = build_comparison_prompt(
        prompt=eval_input.prompt,
        response_a_label=label_a,
        response_a_text=resp_a.response,
        response_b_label=label_b,
        response_b_text=resp_b.response,
        task_type=eval_input.task_type,
        reference_answer=eval_input.reference_answer,
        source_document=eval_input.source_document,
    )

    raw = _call_judge(prompt_text)

    margin = raw.get("margin", "slight")
    if margin not in ("clear", "slight", "tie"):
        margin = "slight"

    winner = raw.get("winner", "tie")

    return PairwiseResult(
        response_a=label_a,
        response_b=label_b,
        winner=winner,
        margin=margin,
        reasoning=raw.get("reasoning", ""),
    )


def run_pairwise_comparison(
    eval_input: EvalInput,
    evaluations: list[ResponseEval],
) -> tuple[list[PairwiseResult], list[str]]:
    labels = [r.label for r in eval_input.responses]
    pairs = list(itertools.combinations(labels, 2))

    pairwise_results: list[PairwiseResult] = []
    wins: dict[str, int] = defaultdict(int)

    for label_a, label_b in pairs:
        logger.info("Pairwise comparison: %s vs %s", label_a, label_b)
        result = _pairwise(eval_input, label_a, label_b)
        pairwise_results.append(result)
        if result.winner != "tie":
            wins[result.winner] += 1

    score_map = {e.label: e.overall_score for e in evaluations}
    ranking = sorted(labels, key=lambda l: (wins[l], score_map.get(l, 0.0)), reverse=True)

    return pairwise_results, ranking


def attach_pairwise_to_report(report: EvalReport, eval_input: EvalInput) -> EvalReport:
    pairwise_results, ranking = run_pairwise_comparison(eval_input, report.evaluations)

    winner = ranking[0] if ranking else None
    winner_eval = next((e for e in report.evaluations if e.label == winner), None)
    winner_reasoning = winner_eval.one_line_verdict if winner_eval else ""

    return EvalReport(
        task_type=report.task_type,
        eval_mode=report.eval_mode,
        timestamp=report.timestamp,
        evaluations=report.evaluations,
        ranking=ranking,
        winner=winner,
        winner_reasoning=winner_reasoning,
        pairwise=pairwise_results,
        key_insight=report.key_insight,
        recommendation=report.recommendation,
        efficiency_note=report.efficiency_note,
    )
