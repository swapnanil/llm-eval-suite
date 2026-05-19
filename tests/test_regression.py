from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.models import (
    DimensionEval,
    EvalReport,
    RegressionCheck,
    RegressionResult,
    ResponseEval,
)
from agent.regression import (
    DEFAULT_THRESHOLD,
    load_baseline,
    run_regression,
    save_baseline,
)


def _make_eval_report(scores: dict[str, float], label: str = "A") -> EvalReport:
    dims = [
        DimensionEval(dimension=dim, score=score, reasoning="test", quote=None)
        for dim, score in scores.items()
    ]
    ev = ResponseEval(
        label=label,
        overall_score=sum(scores.values()) / len(scores),
        dimensions=dims,
        strengths=[],
        weaknesses=[],
        one_line_verdict="test",
    )
    return EvalReport(
        task_type="general",
        eval_mode="score",
        timestamp="2026-01-01T00:00:00+00:00",
        evaluations=[ev],
        ranking=None,
        winner=None,
        winner_reasoning=None,
        pairwise=None,
        key_insight="",
        recommendation="",
        efficiency_note=None,
    )


@pytest.fixture
def tmp_baseline_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BASELINE_DIR", str(tmp_path / "baselines"))
    import agent.regression as reg
    reg._BASELINE_DIR = Path(str(tmp_path / "baselines"))
    return tmp_path / "baselines"


class TestSaveLoadBaseline:
    def test_save_creates_file(self, tmp_baseline_dir):
        report = _make_eval_report({"coherence": 7.0})
        record = save_baseline(report, "v1", "initial baseline")
        assert (tmp_baseline_dir / "v1.json").exists()

    def test_save_returns_record_with_id(self, tmp_baseline_dir):
        report = _make_eval_report({"coherence": 7.0})
        record = save_baseline(report, "v1")
        assert record.baseline_id == "v1"
        assert record.created_at

    def test_load_returns_saved_report(self, tmp_baseline_dir):
        report = _make_eval_report({"coherence": 7.0})
        save_baseline(report, "v1")
        loaded = load_baseline("v1")
        assert loaded.baseline_id == "v1"
        assert loaded.report.task_type == "general"

    def test_load_missing_raises_file_not_found(self, tmp_baseline_dir):
        with pytest.raises(FileNotFoundError, match="no-such-baseline"):
            load_baseline("no-such-baseline")

    def test_baseline_id_slash_sanitized(self, tmp_baseline_dir):
        report = _make_eval_report({"coherence": 7.0})
        save_baseline(report, "prod/v2")
        loaded = load_baseline("prod/v2")
        assert loaded.baseline_id == "prod/v2"


class TestRunRegression:
    def test_no_regression_when_scores_equal(self, tmp_baseline_dir):
        baseline = _make_eval_report({"coherence": 7.0, "conciseness": 8.0})
        save_baseline(baseline, "base")
        new = _make_eval_report({"coherence": 7.0, "conciseness": 8.0})
        result = run_regression(new, "base")
        assert result.passed_overall is True
        assert result.regressions == []

    def test_regression_detected_when_score_drops(self, tmp_baseline_dir):
        baseline = _make_eval_report({"coherence": 8.0})
        save_baseline(baseline, "base")
        new = _make_eval_report({"coherence": 5.0})
        result = run_regression(new, "base")
        assert result.passed_overall is False
        assert len(result.regressions) == 1
        assert "coherence" in result.regressions[0]

    def test_default_threshold_applied(self, tmp_baseline_dir):
        baseline = _make_eval_report({"coherence": 8.0})
        save_baseline(baseline, "base")
        # Drop of exactly DEFAULT_THRESHOLD should pass (delta >= -threshold)
        new = _make_eval_report({"coherence": 8.0 - DEFAULT_THRESHOLD})
        result = run_regression(new, "base")
        assert result.passed_overall is True

    def test_custom_threshold_respected(self, tmp_baseline_dir):
        baseline = _make_eval_report({"coherence": 8.0})
        save_baseline(baseline, "base")
        new = _make_eval_report({"coherence": 7.6})
        result = run_regression(new, "base", thresholds={"coherence": 0.3})
        assert result.passed_overall is False

    def test_improvement_tracked(self, tmp_baseline_dir):
        baseline = _make_eval_report({"coherence": 6.0})
        save_baseline(baseline, "base")
        new = _make_eval_report({"coherence": 8.5})
        result = run_regression(new, "base")
        assert result.passed_overall is True
        assert any("coherence" in i for i in result.improvements)

    def test_summary_mentions_baseline_id(self, tmp_baseline_dir):
        baseline = _make_eval_report({"coherence": 7.0})
        save_baseline(baseline, "my-baseline")
        new = _make_eval_report({"coherence": 7.0})
        result = run_regression(new, "my-baseline")
        assert "my-baseline" in result.summary

    def test_regression_check_fields(self, tmp_baseline_dir):
        baseline = _make_eval_report({"coherence": 7.0})
        save_baseline(baseline, "base")
        new = _make_eval_report({"coherence": 6.0})
        result = run_regression(new, "base")
        check = result.checks[0]
        assert check.dimension == "coherence"
        assert check.baseline_score == pytest.approx(7.0, abs=0.01)
        assert check.new_score == pytest.approx(6.0, abs=0.01)
        assert check.delta == pytest.approx(-1.0, abs=0.01)
        assert check.passed is False

    def test_multiple_dimensions_partial_regression(self, tmp_baseline_dir):
        baseline = _make_eval_report({"coherence": 8.0, "conciseness": 7.0})
        save_baseline(baseline, "base")
        new = _make_eval_report({"coherence": 5.0, "conciseness": 7.5})
        result = run_regression(new, "base")
        assert result.passed_overall is False
        assert len(result.regressions) == 1
        assert "coherence" in result.regressions[0]
