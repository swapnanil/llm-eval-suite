from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.models import EvalInput, LLMResponse
from agent.prompts import TASK_DIMENSIONS, get_dimensions_for_task


class TestRagPreset:
    def test_rag_task_type_accepted_by_eval_input(self):
        ei = EvalInput(
            prompt="What is the capital of France?",
            task_type="rag",
            responses=[LLMResponse(label="A", response="Paris.")],
        )
        assert ei.task_type == "rag"

    def test_rag_dimensions_present_in_task_dimensions(self):
        assert "rag" in TASK_DIMENSIONS

    def test_rag_has_four_ragas_dimensions(self):
        dims = TASK_DIMENSIONS["rag"]
        assert set(dims.keys()) == {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }

    def test_rag_weights_sum_to_one(self):
        dims = TASK_DIMENSIONS["rag"]
        assert sum(dims.values()) == pytest.approx(1.0, abs=1e-6)

    def test_rag_weights_equal(self):
        dims = TASK_DIMENSIONS["rag"]
        weights = list(dims.values())
        assert all(abs(w - 0.25) < 1e-6 for w in weights)

    def test_get_dimensions_for_task_returns_rag_dims(self):
        dims = get_dimensions_for_task("rag")
        assert "answer_relevancy" in dims
        assert "context_precision" in dims
        assert "context_recall" in dims
        assert "faithfulness" in dims

    def test_rag_dimension_descriptions_exist(self):
        from agent.prompts import DIMENSION_DESCRIPTIONS
        for dim in ("answer_relevancy", "context_precision", "context_recall"):
            assert dim in DIMENSION_DESCRIPTIONS
            assert len(DIMENSION_DESCRIPTIONS[dim]) > 20

    def test_rag_description_mentions_ragas(self):
        from agent.prompts import DIMENSION_DESCRIPTIONS
        ragas_dims = ("answer_relevancy", "context_precision", "context_recall")
        for dim in ragas_dims:
            assert "RAGAS" in DIMENSION_DESCRIPTIONS[dim]

    def test_unknown_task_type_falls_back_to_general(self):
        dims = get_dimensions_for_task("nonexistent_task")
        general = get_dimensions_for_task("general")
        assert dims == general

    def test_rag_eval_input_requires_source_document_warning(self, caplog):
        import logging
        ei = EvalInput(
            prompt="What is RAG?",
            task_type="rag",
            responses=[LLMResponse(label="A", response="RAG stands for Retrieval Augmented Generation.")],
        )
        from agent.evaluator import evaluate

        _raw = {
            "label": "A",
            "overall_score": 7.0,
            "dimensions": [
                {"dimension": "faithfulness", "score": 7.0, "reasoning": "ok", "quote": None},
                {"dimension": "answer_relevancy", "score": 7.0, "reasoning": "ok", "quote": None},
                {"dimension": "context_precision", "score": 7.0, "reasoning": "ok", "quote": None},
                {"dimension": "context_recall", "score": 7.0, "reasoning": "ok", "quote": None},
            ],
            "strengths": [],
            "weaknesses": [],
            "one_line_verdict": "ok",
        }
        with patch("agent.evaluator._call_judge", return_value=_raw), caplog.at_level(logging.WARNING):
            evaluate(ei)
        assert any("faithfulness" in r.message or "reference_answer" in r.message for r in caplog.records)
