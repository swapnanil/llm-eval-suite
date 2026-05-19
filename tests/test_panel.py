from __future__ import annotations

from unittest.mock import call, patch

import pytest

from agent.evaluator import evaluate_with_panel
from agent.models import (
    DimensionEval,
    EvalInput,
    LLMResponse,
    PanelEvalReport,
    ResponseEval,
)


def _make_eval_input(num_responses: int = 1) -> EvalInput:
    return EvalInput(
        prompt="Explain why the sky is blue.",
        task_type="general",
        responses=[
            LLMResponse(label=f"R{i+1}", response=f"Response {i+1} text.")
            for i in range(num_responses)
        ],
        eval_mode="score",
    )


def _make_response_eval(label: str, overall: float, dims: dict[str, float]) -> ResponseEval:
    return ResponseEval(
        label=label,
        overall_score=overall,
        dimensions=[
            DimensionEval(dimension=d, score=s, reasoning="test", quote=None)
            for d, s in dims.items()
        ],
        strengths=[],
        weaknesses=[],
        one_line_verdict="test",
    )


_GENERAL_DIMS = {"instruction_following": 7.0, "coherence": 7.0, "conciseness": 7.0}


class TestEvaluateWithPanel:
    def test_returns_panel_eval_report(self):
        evals = [_make_response_eval("R1", 7.0, _GENERAL_DIMS)] * 3
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=3)
        assert isinstance(result, PanelEvalReport)

    def test_num_judges_calls_score_per_judge(self):
        evals = [_make_response_eval("R1", 7.0, _GENERAL_DIMS)] * 3
        with patch("agent.evaluator._score_single_response", side_effect=evals) as mock_score:
            evaluate_with_panel(_make_eval_input(), num_judges=3)
        assert mock_score.call_count == 3

    def test_num_judges_2_accepted(self):
        evals = [_make_response_eval("R1", 7.0, _GENERAL_DIMS)] * 2
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=2)
        assert result.num_judges == 2

    def test_num_judges_5_accepted(self):
        evals = [_make_response_eval("R1", 7.0, _GENERAL_DIMS)] * 5
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=5)
        assert result.num_judges == 5

    def test_num_judges_below_2_raises(self):
        with pytest.raises(ValueError, match="num_judges must be between"):
            evaluate_with_panel(_make_eval_input(), num_judges=1)

    def test_num_judges_above_5_raises(self):
        with pytest.raises(ValueError, match="num_judges must be between"):
            evaluate_with_panel(_make_eval_input(), num_judges=6)

    def test_consensus_score_is_mean_of_dimension_means(self):
        # All judges give uniform scores → consensus = 7.0
        evals = [_make_response_eval("R1", 7.0, _GENERAL_DIMS)] * 3
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=3)
        assert result.response_consensus[0].consensus_score == pytest.approx(7.0, abs=0.01)

    def test_disagreement_flag_set_when_variance_high(self):
        # coherence: 2.0 then 9.0 then 9.0 → variance ≈ 16.3 (> 1.5)
        evals = [
            _make_response_eval("R1", 2.0, {"instruction_following": 7.0, "coherence": 2.0, "conciseness": 7.0}),
            _make_response_eval("R1", 9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
            _make_response_eval("R1", 9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
        ]
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=3)
        rc = result.response_consensus[0]
        coherence_dim = next(d for d in rc.dimension_consensus if d.dimension == "coherence")
        assert coherence_dim.disagreement_flag is True

    def test_needs_human_review_true_when_disagreement(self):
        evals = [
            _make_response_eval("R1", 2.0, {"instruction_following": 7.0, "coherence": 2.0, "conciseness": 7.0}),
            _make_response_eval("R1", 9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
            _make_response_eval("R1", 9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
        ]
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=3)
        assert result.response_consensus[0].needs_human_review is True

    def test_needs_human_review_false_when_stable(self):
        evals = [_make_response_eval("R1", 7.0, _GENERAL_DIMS)] * 3
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=3)
        assert result.response_consensus[0].needs_human_review is False
        assert result.response_consensus[0].reviewer_note is None

    def test_winner_is_highest_consensus_response(self):
        # R1 judges score 6.0, R2 judges score 9.0
        evals = (
            [_make_response_eval("R1", 6.0, {"instruction_following": 6.0, "coherence": 6.0, "conciseness": 6.0})] * 3
            + [_make_response_eval("R2", 9.0, {"instruction_following": 9.0, "coherence": 9.0, "conciseness": 9.0})] * 3
        )
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(num_responses=2), num_judges=3)
        assert result.winner == "R2"

    def test_flagged_for_review_populated(self):
        evals = [
            _make_response_eval("R1", 2.0, {"instruction_following": 7.0, "coherence": 2.0, "conciseness": 7.0}),
            _make_response_eval("R1", 9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
            _make_response_eval("R1", 9.0, {"instruction_following": 7.0, "coherence": 9.0, "conciseness": 7.0}),
        ]
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=3)
        assert "R1" in result.flagged_for_review

    def test_panel_key_insight_mentions_winner(self):
        evals = (
            [_make_response_eval("R1", 5.0, {"instruction_following": 5.0, "coherence": 5.0, "conciseness": 5.0})] * 2
            + [_make_response_eval("R2", 9.0, {"instruction_following": 9.0, "coherence": 9.0, "conciseness": 9.0})] * 2
        )
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(num_responses=2), num_judges=2)
        assert "R2" in result.panel_key_insight

    def test_timestamp_is_set(self):
        evals = [_make_response_eval("R1", 7.0, _GENERAL_DIMS)] * 2
        with patch("agent.evaluator._score_single_response", side_effect=evals):
            result = evaluate_with_panel(_make_eval_input(), num_judges=2)
        assert result.timestamp
