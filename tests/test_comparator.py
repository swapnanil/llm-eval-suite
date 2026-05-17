from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.comparator import attach_pairwise_to_report, run_pairwise_comparison
from agent.models import (
    DimensionEval,
    EvalInput,
    EvalReport,
    LLMResponse,
    PairwiseResult,
    ResponseEval,
)
from datetime import datetime, timezone


def _make_eval_input(labels: list[str]) -> EvalInput:
    return EvalInput(
        prompt="Summarise the key points.",
        task_type="summarisation",
        responses=[LLMResponse(label=lbl, response=f"Response from {lbl}.") for lbl in labels],
        eval_mode="rank",
        source_document="Source text here.",
    )


def _make_response_eval(label: str, score: float) -> ResponseEval:
    return ResponseEval(
        label=label,
        overall_score=score,
        dimensions=[
            DimensionEval(dimension="instruction_following", score=score, reasoning="test", quote=None)
        ],
        strengths=["good"],
        weaknesses=[],
        one_line_verdict=f"{label} verdict",
    )


def _make_report(evaluations: list[ResponseEval]) -> EvalReport:
    return EvalReport(
        task_type="summarisation",
        eval_mode="rank",
        timestamp=datetime.now(timezone.utc).isoformat(),
        evaluations=evaluations,
        key_insight="Test insight",
        recommendation="Test recommendation",
    )


def _pairwise_judge(winner: str, margin: str = "clear") -> dict:
    return {
        "response_a": "A",
        "response_b": "B",
        "winner": winner,
        "margin": margin,
        "reasoning": f"{winner} is better because of specific evidence.",
    }


class TestPairwiseComparison:
    @patch("agent.comparator._call_judge")
    def test_pairwise_returns_correct_structure(self, mock_judge):
        mock_judge.return_value = _pairwise_judge("A")
        eval_input = _make_eval_input(["A", "B"])
        evals = [_make_response_eval("A", 8.0), _make_response_eval("B", 6.0)]
        results, ranking = run_pairwise_comparison(eval_input, evals)
        assert len(results) == 1
        assert results[0].winner in ("A", "B", "tie")
        assert ranking[0] in ("A", "B")

    @patch("agent.comparator._call_judge")
    def test_ranking_reflects_win_counts(self, mock_judge):
        mock_judge.side_effect = [
            {"response_a": "A", "response_b": "B", "winner": "A", "margin": "clear", "reasoning": "A is better."},
            {"response_a": "A", "response_b": "C", "winner": "A", "margin": "slight", "reasoning": "A edges C."},
            {"response_a": "B", "response_b": "C", "winner": "B", "margin": "clear", "reasoning": "B is better than C."},
        ]
        eval_input = _make_eval_input(["A", "B", "C"])
        evals = [_make_response_eval(l, s) for l, s in [("A", 8.0), ("B", 7.0), ("C", 5.0)]]
        results, ranking = run_pairwise_comparison(eval_input, evals)
        assert len(results) == 3
        assert ranking[0] == "A"
        assert ranking[-1] == "C"

    @patch("agent.comparator._call_judge")
    def test_tie_margin_accepted(self, mock_judge):
        mock_judge.return_value = {
            "response_a": "A",
            "response_b": "B",
            "winner": "tie",
            "margin": "tie",
            "reasoning": "Genuinely indistinguishable.",
        }
        eval_input = _make_eval_input(["A", "B"])
        evals = [_make_response_eval("A", 7.5), _make_response_eval("B", 7.5)]
        results, _ = run_pairwise_comparison(eval_input, evals)
        assert results[0].winner == "tie"
        assert results[0].margin == "tie"


class TestAttachPairwiseToReport:
    @patch("agent.comparator._call_judge")
    def test_report_gets_pairwise_and_ranking(self, mock_judge):
        mock_judge.return_value = _pairwise_judge("A", "clear")
        eval_input = _make_eval_input(["A", "B"])
        evals = [_make_response_eval("A", 9.0), _make_response_eval("B", 5.0)]
        report = _make_report(evals)
        updated = attach_pairwise_to_report(report, eval_input)

        assert updated.pairwise is not None
        assert len(updated.pairwise) == 1
        assert updated.ranking is not None
        assert updated.winner is not None

    @patch("agent.comparator._call_judge")
    def test_winner_is_in_evaluations(self, mock_judge):
        mock_judge.return_value = _pairwise_judge("B", "slight")
        eval_input = _make_eval_input(["A", "B"])
        evals = [_make_response_eval("A", 7.0), _make_response_eval("B", 8.0)]
        report = _make_report(evals)
        updated = attach_pairwise_to_report(report, eval_input)

        all_labels = [e.label for e in updated.evaluations]
        assert updated.winner in all_labels
