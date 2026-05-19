from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .models import BaselineRecord, EvalReport, RegressionCheck, RegressionResult

DEFAULT_THRESHOLD = 0.5
_BASELINE_DIR = Path(os.getenv("BASELINE_DIR", "./baselines"))


def _baseline_path(baseline_id: str) -> Path:
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = baseline_id.replace("/", "_").replace("\\", "_")
    return _BASELINE_DIR / f"{safe_id}.json"


def save_baseline(
    report: EvalReport,
    baseline_id: str,
    description: str = "",
) -> BaselineRecord:
    record = BaselineRecord(
        baseline_id=baseline_id,
        description=description,
        created_at=datetime.now(timezone.utc).isoformat(),
        report=report,
    )
    path = _baseline_path(baseline_id)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def load_baseline(baseline_id: str) -> BaselineRecord:
    path = _baseline_path(baseline_id)
    if not path.exists():
        raise FileNotFoundError(f"No baseline found with id '{baseline_id}' at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return BaselineRecord(**data)


def list_baselines() -> list[str]:
    if not _BASELINE_DIR.exists():
        return []
    return [p.stem for p in _BASELINE_DIR.glob("*.json")]


def run_regression(
    new_report: EvalReport,
    baseline_id: str,
    thresholds: dict[str, float] | None = None,
) -> RegressionResult:
    baseline_record = load_baseline(baseline_id)
    baseline_report = baseline_record.report

    # Build dimension score maps from both reports (average across all responses)
    def _dim_scores(report: EvalReport) -> dict[str, float]:
        scores: dict[str, list[float]] = {}
        for ev in report.evaluations:
            for dim in ev.dimensions:
                scores.setdefault(dim.dimension, []).append(dim.score)
        return {d: sum(vals) / len(vals) for d, vals in scores.items()}

    baseline_scores = _dim_scores(baseline_report)
    new_scores = _dim_scores(new_report)

    all_dims = set(baseline_scores) | set(new_scores)
    checks: list[RegressionCheck] = []
    regressions: list[str] = []
    improvements: list[str] = []

    for dim in sorted(all_dims):
        baseline_s = baseline_scores.get(dim, 0.0)
        new_s = new_scores.get(dim, 0.0)
        delta = new_s - baseline_s
        threshold = (thresholds or {}).get(dim, DEFAULT_THRESHOLD)
        passed = delta >= -threshold
        checks.append(RegressionCheck(
            dimension=dim,
            baseline_score=round(baseline_s, 3),
            new_score=round(new_s, 3),
            delta=round(delta, 3),
            passed=passed,
            threshold_used=threshold,
        ))
        if not passed:
            regressions.append(f"{dim}: dropped {abs(delta):.2f} (threshold {threshold})")
        elif delta > 0.2:
            improvements.append(f"{dim}: improved +{delta:.2f}")

    passed_overall = all(c.passed for c in checks)
    if passed_overall:
        summary = f"All {len(checks)} dimension checks passed against baseline '{baseline_id}'."
    else:
        summary = (
            f"{len(regressions)} regression(s) detected vs baseline '{baseline_id}': "
            + "; ".join(regressions)
        )

    return RegressionResult(
        baseline_id=baseline_id,
        passed_overall=passed_overall,
        checks=checks,
        regressions=regressions,
        improvements=improvements,
        summary=summary,
    )
