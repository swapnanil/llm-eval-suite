from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.models import (
    DimensionEval,
    EvalInput,
    LLMResponse,
    PromptVariant,
    ResponseEval,
    SensitivityReport,
)
from agent.sensitivity import (
    _SENSITIVITY_THRESHOLD_HIGH,
    _SENSITIVITY_THRESHOLD_MODERATE,
    analyse_prompt_sensitivity,
)

FIXED_RESPONSE = "The product increases efficiency by automating repetitive tasks."

PROMPTS_3 = [
    PromptVariant(label="Formal", prompt="Summarise the benefits of this product formally."),
    PromptVariant(label="Casual", prompt="What does this product do?"),
    PromptVariant(label="Technical", prompt="Describe the product's efficiency gains."),
]


def _make_response_eval(overall: float, dims: dict[str, float]) -> ResponseEval:
    return ResponseEval(
        label="response",
        overall_score=overall,
        dimensions=[
            DimensionEval(dimension=d, score=s, reasoning="test", quote=None)
            for d, s in dims.items()
        ],
        strengths=[],
        weaknesses=[],
        one_line_verdict="test",
    )


def _mock_score(side_effects: list[ResponseEval]):
    return patch(
        "agent.sensitivity._score_single_response",
        side_effect=side_effects,
    )


class TestAnalysePromptSensitivity:
    def test_returns_sensitivity_report(self):
        evals = [
            _make_response_eval(7.0, {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}),
            _make_response_eval(7.0, {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}),
            _make_response_eval(7.0, {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3, FIXED_RESPONSE, "general")
        assert isinstance(result, SensitivityReport)

    def test_requires_at_least_2_prompts(self):
        with pytest.raises(ValueError, match="At least 2"):
            analyse_prompt_sensitivity([PROMPTS_3[0]], FIXED_RESPONSE, "general")

    def test_rejects_more_than_5_prompts(self):
        six_prompts = [PromptVariant(label=f"P{i}", prompt=f"Prompt {i}") for i in range(6)]
        with pytest.raises(ValueError, match="Maximum 5"):
            analyse_prompt_sensitivity(six_prompts, FIXED_RESPONSE, "general")

    def test_recommended_prompt_is_highest_scoring(self):
        evals = [
            _make_response_eval(5.0, {"instruction_following": 5.0, "coherence": 5.0, "conciseness": 5.0}),
            _make_response_eval(9.0, {"instruction_following": 9.0, "coherence": 9.0, "conciseness": 9.0}),
            _make_response_eval(7.0, {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3, FIXED_RESPONSE, "general")
        assert result.recommended_prompt == "Casual"

    def test_stable_dimensions_when_scores_equal(self):
        evals = [
            _make_response_eval(7.0, {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}),
            _make_response_eval(7.0, {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3[:2], FIXED_RESPONSE, "general")
        for d in result.dimension_sensitivities:
            assert d.sensitivity == "stable"
            assert d.variance == 0.0

    def test_sensitive_dimensions_when_high_variance(self):
        # coherence: 2.0 vs 9.0 → variance = 24.5 (well above 1.5)
        evals = [
            _make_response_eval(2.0, {"instruction_following": 7.0, "coherence": 2.0, "conciseness": 7.0}),
            _make_response_eval(9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3[:2], FIXED_RESPONSE, "general")
        coherence = next(d for d in result.dimension_sensitivities if d.dimension == "coherence")
        assert coherence.sensitivity == "sensitive"

    def test_most_sensitive_dimensions_populated(self):
        evals = [
            _make_response_eval(2.0, {"instruction_following": 7.0, "coherence": 2.0, "conciseness": 7.0}),
            _make_response_eval(9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3[:2], FIXED_RESPONSE, "general")
        assert "coherence" in result.most_sensitive_dimensions

    def test_stability_summary_stable_when_all_stable(self):
        evals = [
            _make_response_eval(7.0, {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}),
            _make_response_eval(7.1, {"instruction_following": 7.1, "coherence": 7.0, "conciseness": 7.1}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3[:2], FIXED_RESPONSE, "general")
        assert "stable" in result.stability_summary.lower()

    def test_stability_summary_flags_sensitive_dimensions(self):
        evals = [
            _make_response_eval(2.0, {"instruction_following": 7.0, "coherence": 2.0, "conciseness": 7.0}),
            _make_response_eval(9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3[:2], FIXED_RESPONSE, "general")
        assert "coherence" in result.stability_summary

    def test_prompt_scores_length_matches_variants(self):
        evals = [
            _make_response_eval(6.0, {"instruction_following": 6.0, "coherence": 6.0, "conciseness": 6.0}),
            _make_response_eval(7.0, {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}),
            _make_response_eval(8.0, {"instruction_following": 8.0, "coherence": 8.0, "conciseness": 8.0}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3, FIXED_RESPONSE, "general")
        assert len(result.prompt_scores) == 3

    def test_dimension_mean_computed_correctly(self):
        evals = [
            _make_response_eval(4.0, {"instruction_following": 4.0, "coherence": 4.0, "conciseness": 4.0}),
            _make_response_eval(8.0, {"instruction_following": 8.0, "coherence": 8.0, "conciseness": 8.0}),
        ]
        with _mock_score(evals):
            result = analyse_prompt_sensitivity(PROMPTS_3[:2], FIXED_RESPONSE, "general")
        coherence = next(d for d in result.dimension_sensitivities if d.dimension == "coherence")
        assert coherence.mean_score == pytest.approx(6.0, abs=0.01)
