from __future__ import annotations

import statistics

from .evaluator import _score_single_response
from .models import (
    DimensionSensitivity,
    EvalInput,
    LLMResponse,
    PromptScore,
    PromptVariant,
    SensitivityReport,
)
from .prompts import get_dimensions_for_task

_SENSITIVITY_THRESHOLD_MODERATE = 0.5
_SENSITIVITY_THRESHOLD_HIGH = 1.5


def analyse_prompt_sensitivity(
    prompts: list[PromptVariant],
    fixed_response: str,
    task_type: str,
    fixed_response_label: str = "response",
) -> SensitivityReport:
    if len(prompts) < 2:
        raise ValueError("At least 2 prompt variants are required for sensitivity analysis.")
    if len(prompts) > 5:
        raise ValueError("Maximum 5 prompt variants allowed.")

    dimensions = get_dimensions_for_task(task_type)

    prompt_scores: list[PromptScore] = []

    for variant in prompts:
        eval_input = EvalInput(
            prompt=variant.prompt,
            task_type=task_type,  # type: ignore[arg-type]
            responses=[LLMResponse(label=fixed_response_label, response=fixed_response)],
        )
        response_eval = _score_single_response(eval_input, 0, dimensions)

        dim_scores = {d.dimension: d.score for d in response_eval.dimensions}
        prompt_scores.append(PromptScore(
            label=variant.label,
            overall_score=response_eval.overall_score,
            dimension_scores=dim_scores,
        ))

    # Compute variance per dimension
    all_dims = list(dimensions.keys())
    dim_sensitivities: list[DimensionSensitivity] = []

    for dim in all_dims:
        scores = [ps.dimension_scores.get(dim, 5.0) for ps in prompt_scores]
        mean = statistics.mean(scores)
        variance = statistics.variance(scores) if len(scores) > 1 else 0.0

        if variance >= _SENSITIVITY_THRESHOLD_HIGH:
            sensitivity = "sensitive"
        elif variance >= _SENSITIVITY_THRESHOLD_MODERATE:
            sensitivity = "moderate"
        else:
            sensitivity = "stable"

        dim_sensitivities.append(DimensionSensitivity(
            dimension=dim,
            mean_score=round(mean, 3),
            variance=round(variance, 3),
            sensitivity=sensitivity,  # type: ignore[arg-type]
        ))

    most_sensitive = [
        d.dimension for d in dim_sensitivities if d.sensitivity == "sensitive"
    ]

    # Recommend the prompt with the highest overall score
    best = max(prompt_scores, key=lambda ps: ps.overall_score)

    if most_sensitive:
        stability_note = (
            f"High sensitivity detected in: {', '.join(most_sensitive)}. "
            "Small prompt changes cause significant score variation in these dimensions — review wording carefully."
        )
    else:
        stable_count = sum(1 for d in dim_sensitivities if d.sensitivity == "stable")
        stability_note = (
            f"Scores are stable across prompt variants ({stable_count}/{len(dim_sensitivities)} dimensions stable). "
            "Prompt phrasing has low impact on evaluation outcomes."
        )

    return SensitivityReport(
        recommended_prompt=best.label,
        prompt_scores=prompt_scores,
        dimension_sensitivities=dim_sensitivities,
        most_sensitive_dimensions=most_sensitive,
        stability_summary=stability_note,
    )
