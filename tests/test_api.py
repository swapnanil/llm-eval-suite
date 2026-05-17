from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agent.models import (
    DimensionEval,
    EvalReport,
    LLMResponse,
    PairwiseResult,
    ResponseEval,
)

# ---- Fixtures ----

@pytest.fixture
def client():
    from api import app
    return TestClient(app)


def _make_response_eval(label: str, score: float) -> ResponseEval:
    return ResponseEval(
        label=label,
        overall_score=score,
        dimensions=[
            DimensionEval(dimension="instruction_following", score=score, reasoning="test", quote=None),
            DimensionEval(dimension="faithfulness", score=max(1.0, score - 0.5), reasoning="test", quote=None),
        ],
        strengths=["clear"],
        weaknesses=["could improve"],
        one_line_verdict=f"{label} scored {score}.",
    )


def _make_report(labels_scores: list[tuple[str, float]], mode: str = "rank") -> EvalReport:
    evals = [_make_response_eval(l, s) for l, s in labels_scores]
    ranking = [l for l, _ in sorted(labels_scores, key=lambda x: x[1], reverse=True)]
    winner = ranking[0] if ranking else None
    return EvalReport(
        task_type="general",
        eval_mode=mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        evaluations=evals,
        ranking=ranking,
        winner=winner,
        winner_reasoning=f"{winner} is best." if winner else None,
        pairwise=None,
        key_insight="Key finding.",
        recommendation="Use the winner.",
    )


# ---- Health and dimensions ----

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data


class TestDimensions:
    def test_dimensions_returns_all_task_types(self, client):
        resp = client.get("/dimensions")
        assert resp.status_code == 200
        data = resp.json()
        for task in ("summarisation", "qa", "code_generation", "creative_writing", "general"):
            assert task in data

    def test_each_dimension_has_weight_and_description(self, client):
        resp = client.get("/dimensions")
        data = resp.json()
        for task, dims in data.items():
            for dim_name, dim_info in dims.items():
                assert "weight" in dim_info, f"Missing weight for {task}/{dim_name}"
                assert "description" in dim_info, f"Missing description for {task}/{dim_name}"


# ---- POST /eval ----

class TestEvalEndpoint:
    @patch("api.evaluate")
    @patch("api.attach_pairwise_to_report")
    def test_score_mode_single_response(self, mock_pairwise, mock_evaluate, client):
        mock_evaluate.return_value = _make_report([("A", 8.0)], mode="score")
        payload = {
            "prompt": "Summarise this.",
            "task_type": "general",
            "responses": [{"label": "A", "response": "A response."}],
            "eval_mode": "score",
        }
        resp = client.post("/eval", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["eval_mode"] == "score"
        assert len(data["evaluations"]) == 1

    @patch("api.evaluate")
    @patch("api.attach_pairwise_to_report")
    def test_rank_mode_calls_pairwise(self, mock_pairwise, mock_evaluate, client):
        base_report = _make_report([("A", 9.0), ("B", 6.0)], mode="rank")
        mock_evaluate.return_value = base_report
        mock_pairwise.return_value = base_report
        payload = {
            "prompt": "Write a summary.",
            "task_type": "summarisation",
            "responses": [
                {"label": "A", "response": "First response."},
                {"label": "B", "response": "Second response."},
            ],
            "eval_mode": "rank",
        }
        resp = client.post("/eval", json=payload)
        assert resp.status_code == 200
        mock_pairwise.assert_called_once()

    def test_empty_response_text_rejected(self, client):
        payload = {
            "prompt": "Test",
            "task_type": "general",
            "responses": [{"label": "A", "response": "  "}],
            "eval_mode": "score",
        }
        resp = client.post("/eval", json=payload)
        assert resp.status_code == 422

    def test_too_many_responses_rejected(self, client):
        payload = {
            "prompt": "Test",
            "task_type": "general",
            "responses": [{"label": str(i), "response": "text"} for i in range(11)],
            "eval_mode": "score",
        }
        resp = client.post("/eval", json=payload)
        assert resp.status_code == 422


# ---- POST /eval/quick ----

class TestEvalQuick:
    @patch("api.evaluate")
    @patch("api.attach_pairwise_to_report")
    def test_quick_eval_single_response(self, mock_pairwise, mock_evaluate, client):
        mock_evaluate.return_value = _make_report([("Model A", 7.5)], mode="score")
        payload = {
            "prompt": "Classify this sentiment.",
            "responses": [{"label": "Model A", "response": "Positive sentiment."}],
            "task_type": "classification",
        }
        resp = client.post("/eval/quick", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "evaluations" in data


# ---- POST /eval/batch ----

class TestEvalBatch:
    @patch("api.evaluate")
    @patch("api.attach_pairwise_to_report")
    def test_batch_returns_same_count_as_inputs(self, mock_pairwise, mock_evaluate, client):
        mock_evaluate.return_value = _make_report([("A", 8.0)], mode="score")
        mock_pairwise.return_value = _make_report([("A", 8.0)], mode="score")

        inputs = [
            {
                "prompt": f"Prompt {i}",
                "task_type": "general",
                "responses": [{"label": "A", "response": f"Response {i}"}],
                "eval_mode": "score",
            }
            for i in range(4)
        ]
        resp = client.post("/eval/batch", json=inputs)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_evals"] == 4
        assert len(data["reports"]) == 4

    @patch("api.evaluate")
    @patch("api.attach_pairwise_to_report")
    def test_batch_with_5_inputs(self, mock_pairwise, mock_evaluate, client):
        mock_evaluate.return_value = _make_report([("A", 7.0)], mode="score")
        mock_pairwise.return_value = _make_report([("A", 7.0)], mode="score")

        inputs = [
            {"prompt": f"P{i}", "task_type": "general", "responses": [{"label": "A", "response": f"R{i}"}], "eval_mode": "score"}
            for i in range(5)
        ]
        resp = client.post("/eval/batch", json=inputs)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_evals"] == 5

    def test_batch_empty_input_rejected(self, client):
        resp = client.post("/eval/batch", json=[])
        assert resp.status_code == 422


# ---- JUnit XML format ----

class TestJUnitOutput:
    def test_junit_is_valid_xml(self):
        from main import _to_junit
        report = _make_report([("A", 8.5), ("B", 5.0)], mode="rank")
        junit_str = _to_junit(report)
        tree = ET.fromstring(junit_str.split("?>", 1)[-1].strip())
        assert tree.tag == "testsuite"

    def test_junit_failures_below_threshold(self):
        import os
        os.environ["EVAL_PASS_THRESHOLD"] = "7.0"
        from main import _to_junit
        report = _make_report([("A", 8.5), ("B", 4.0)], mode="rank")
        junit_str = _to_junit(report)
        tree = ET.fromstring(junit_str.split("?>", 1)[-1].strip())
        failures = int(tree.get("failures", "0"))
        assert failures > 0

    def test_junit_all_pass_when_scores_above_threshold(self):
        import os
        os.environ["EVAL_PASS_THRESHOLD"] = "5.0"
        from main import _to_junit
        report = _make_report([("A", 8.5), ("B", 7.0)], mode="rank")
        junit_str = _to_junit(report)
        tree = ET.fromstring(junit_str.split("?>", 1)[-1].strip())
        failures = int(tree.get("failures", "0"))
        assert failures == 0
