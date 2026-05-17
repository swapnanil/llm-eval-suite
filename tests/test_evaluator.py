from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.evaluator import _clamp_score, _compute_weighted_score, evaluate
from agent.models import DimensionEval, EvalInput, LLMResponse, ResponseEval


def _make_eval_input(**kwargs) -> EvalInput:
    defaults = dict(
        prompt="Summarise this.",
        task_type="general",
        responses=[LLMResponse(label="A", response="Some response text.")],
        eval_mode="score",
    )
    defaults.update(kwargs)
    return EvalInput(**defaults)


def _make_response_eval(label: str, overall: float, dimensions: list[tuple[str, float]]) -> ResponseEval:
    return ResponseEval(
        label=label,
        overall_score=overall,
        dimensions=[
            DimensionEval(dimension=dim, score=score, reasoning="test", quote=None)
            for dim, score in dimensions
        ],
        strengths=["strength"],
        weaknesses=["weakness"],
        one_line_verdict="verdict",
    )


def _judge_score_response(label: str, dims: list[tuple[str, float]], overall: float) -> dict:
    return {
        "label": label,
        "overall_score": overall,
        "dimensions": [
            {"dimension": d, "score": s, "reasoning": f"Reasoning for {d}", "quote": None}
            for d, s in dims
        ],
        "strengths": ["clear writing"],
        "weaknesses": ["could be shorter"],
        "one_line_verdict": "Solid response.",
    }


def _judge_rank_response(labels: list[str]) -> dict:
    return {
        "ranking": labels,
        "winner": labels[0],
        "winner_reasoning": f"{labels[0]} is the best.",
        "key_insight": "Key finding.",
        "recommendation": "Use this model.",
    }


# --- Unit tests ---

class TestClampScore:
    def test_valid_score_unchanged(self):
        assert _clamp_score(7.5, "A", "coherence") == 7.5

    def test_score_below_1_clamped(self):
        assert _clamp_score(0.0, "A", "coherence") == 1.0

    def test_score_above_10_clamped(self):
        assert _clamp_score(11.0, "A", "coherence") == 10.0

    def test_boundary_scores(self):
        assert _clamp_score(1.0, "A", "d") == 1.0
        assert _clamp_score(10.0, "A", "d") == 10.0


class TestComputeWeightedScore:
    def test_weighted_average(self):
        dims = [
            DimensionEval(dimension="faithfulness", score=8.0, reasoning="", quote=None),
            DimensionEval(dimension="coherence", score=6.0, reasoning="", quote=None),
        ]
        weights = {"faithfulness": 0.7, "coherence": 0.3}
        result = _compute_weighted_score(dims, weights)
        expected = 8.0 * 0.7 + 6.0 * 0.3
        assert abs(result - expected) < 0.001

    def test_unknown_dimension_has_zero_weight(self):
        dims = [DimensionEval(dimension="unknown", score=9.0, reasoning="", quote=None)]
        weights = {"faithfulness": 1.0}
        result = _compute_weighted_score(dims, weights)
        assert result == 9.0  # falls back to simple average


class TestEvalInputValidation:
    def test_empty_response_text_rejected(self):
        with pytest.raises(Exception):
            EvalInput(
                prompt="Summarise this.",
                task_type="general",
                responses=[LLMResponse(label="A", response="   ")],
                eval_mode="score",
            )

    def test_more_than_10_responses_rejected(self):
        with pytest.raises(Exception):
            EvalInput(
                prompt="Test",
                task_type="general",
                responses=[LLMResponse(label=str(i), response="text") for i in range(11)],
                eval_mode="score",
            )

    def test_zero_responses_rejected(self):
        with pytest.raises(Exception):
            EvalInput(
                prompt="Test",
                task_type="general",
                responses=[],
                eval_mode="score",
            )


# --- Integration tests with mocked judge ---

class TestEvaluateMocked:
    @patch("agent.evaluator._call_judge")
    def test_score_mode_returns_report(self, mock_judge):
        mock_judge.return_value = _judge_score_response(
            "A", [("instruction_following", 8.0), ("coherence", 7.0), ("conciseness", 8.0)], 7.8
        )
        eval_input = _make_eval_input()
        report = evaluate(eval_input)
        assert report.eval_mode == "score"
        assert len(report.evaluations) == 1
        assert report.evaluations[0].label == "A"
        assert 1.0 <= report.evaluations[0].overall_score <= 10.0

    @patch("agent.evaluator._call_judge")
    def test_rank_mode_calls_judge_twice(self, mock_judge):
        mock_judge.side_effect = [
            _judge_score_response("A", [("instruction_following", 8.0), ("coherence", 7.0), ("conciseness", 8.0)], 7.8),
            _judge_score_response("B", [("instruction_following", 6.0), ("coherence", 6.0), ("conciseness", 6.0)], 6.0),
            _judge_rank_response(["A", "B"]),
        ]
        eval_input = _make_eval_input(
            responses=[
                LLMResponse(label="A", response="First response."),
                LLMResponse(label="B", response="Second response."),
            ],
            eval_mode="rank",
        )
        report = evaluate(eval_input)
        assert report.ranking == ["A", "B"]
        assert report.winner == "A"

    @patch("agent.evaluator._call_judge")
    def test_scores_clamped_to_valid_range(self, mock_judge):
        mock_judge.return_value = {
            "label": "A",
            "overall_score": 15.0,
            "dimensions": [
                {"dimension": "instruction_following", "score": -2.0, "reasoning": "bad", "quote": None}
            ],
            "strengths": [],
            "weaknesses": [],
            "one_line_verdict": "test",
        }
        eval_input = _make_eval_input()
        report = evaluate(eval_input)
        assert report.evaluations[0].overall_score <= 10.0
        assert report.evaluations[0].dimensions[0].score >= 1.0


class TestHallucinationDetection:
    @patch("agent.evaluator._call_judge")
    def test_hallucination_gives_low_faithfulness(self, mock_judge):
        mock_judge.side_effect = [
            {
                "label": "Hallucinating Model",
                "overall_score": 3.5,
                "dimensions": [
                    {"dimension": "instruction_following", "score": 6.0, "reasoning": "answered question", "quote": None},
                    {"dimension": "coherence", "score": 7.0, "reasoning": "fluent", "quote": None},
                    {"dimension": "conciseness", "score": 7.0, "reasoning": "brief", "quote": None},
                    {"dimension": "faithfulness", "score": 1.5, "reasoning": "States 30-day policy but source says 14 days — clear hallucination.", "quote": "30-day refund policy on digital products"},
                    {"dimension": "accuracy", "score": 1.0, "reasoning": "Wrong answer.", "quote": None},
                    {"dimension": "appropriate_hedging", "score": 2.0, "reasoning": "No hedging on wrong claim.", "quote": None},
                ],
                "strengths": [],
                "weaknesses": ["Hallucinates refund window"],
                "one_line_verdict": "Confidently wrong.",
            },
            _judge_rank_response(["Correct Model", "Hallucinating Model"]),
        ]

        eval_input_path = Path(__file__).parent.parent / "examples" / "eval_qa.json"
        raw = json.loads(eval_input_path.read_text())
        raw["responses"] = [raw["responses"][1]]
        raw["eval_mode"] = "score"
        eval_input = EvalInput.model_validate(raw)

        report = evaluate(eval_input)
        faithfulness_dim = next(
            d for d in report.evaluations[0].dimensions if d.dimension == "faithfulness"
        )
        assert faithfulness_dim.score <= 3.0, (
            f"Expected faithfulness ≤ 3 for hallucinating response, got {faithfulness_dim.score}"
        )


class TestInstructionViolation:
    @patch("agent.evaluator._call_judge")
    def test_wrong_bullet_count_penalises_instruction_following(self, mock_judge):
        mock_judge.side_effect = [
            {
                "label": "Too Many Bullets",
                "overall_score": 5.0,
                "dimensions": [
                    {"dimension": "instruction_following", "score": 3.0, "reasoning": "Asked for 3 bullets, provided 8 — ignores the count constraint.", "quote": None},
                    {"dimension": "coherence", "score": 8.0, "reasoning": "Well structured.", "quote": None},
                    {"dimension": "conciseness", "score": 4.0, "reasoning": "Too verbose.", "quote": None},
                    {"dimension": "faithfulness", "score": 9.0, "reasoning": "Accurate content.", "quote": None},
                    {"dimension": "coverage", "score": 9.5, "reasoning": "Covers all points.", "quote": None},
                    {"dimension": "compression_quality", "score": 5.0, "reasoning": "Too expanded.", "quote": None},
                ],
                "strengths": ["Accurate content"],
                "weaknesses": ["Ignores bullet count instruction"],
                "one_line_verdict": "Accurate but violates the 3-bullet constraint.",
            },
            _judge_rank_response(["Correct Response", "Too Many Bullets"]),
        ]

        eval_input_path = Path(__file__).parent.parent / "examples" / "eval_summarisation.json"
        raw = json.loads(eval_input_path.read_text())
        raw["responses"] = [raw["responses"][2]]
        raw["eval_mode"] = "score"
        eval_input = EvalInput.model_validate(raw)

        report = evaluate(eval_input)
        if_dim = next(
            d for d in report.evaluations[0].dimensions if d.dimension == "instruction_following"
        )
        assert if_dim.score <= 5.0, (
            f"Expected instruction_following ≤ 5 for wrong bullet count, got {if_dim.score}"
        )
