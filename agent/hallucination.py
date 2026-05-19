from __future__ import annotations

import os

from .evaluator import _call_judge
from .models import HallucinationReport, SuspiciousClaim
from .prompts import HALLUCINATION_SYSTEM_PROMPT, build_hallucination_prompt


def detect_hallucinations(
    response_text: str,
    source: str,
    response_label: str = "response",
) -> HallucinationReport:
    if not response_text.strip():
        raise ValueError("Response text cannot be empty.")
    if not source.strip():
        raise ValueError("Source document cannot be empty.")

    prompt = build_hallucination_prompt(response_label, response_text, source)

    # Temporarily swap the system prompt used by _call_judge
    import agent.prompts as _prompts
    original = _prompts.JUDGE_SYSTEM_PROMPT
    _prompts.JUDGE_SYSTEM_PROMPT = HALLUCINATION_SYSTEM_PROMPT
    try:
        raw = _call_judge(prompt)
    finally:
        _prompts.JUDGE_SYSTEM_PROMPT = original

    claims = []
    for c in raw.get("suspicious_claims", []):
        sev = c.get("severity", "low")
        if sev not in ("critical", "moderate", "low"):
            sev = "low"
        claims.append(SuspiciousClaim(
            claim=c.get("claim", ""),
            severity=sev,  # type: ignore[arg-type]
            is_supported=bool(c.get("is_supported", True)),
            evidence_or_gap=c.get("evidence_or_gap", ""),
        ))

    risk = raw.get("risk_level", "low")
    if risk not in ("none", "low", "moderate", "high", "critical"):
        risk = "moderate"

    rate = float(raw.get("hallucination_rate", 0.0))
    rate = max(0.0, min(1.0, rate))

    safe = bool(raw.get("safe_to_use", risk in ("none", "low")))

    return HallucinationReport(
        response_label=response_label,
        risk_level=risk,  # type: ignore[arg-type]
        hallucination_rate=rate,
        safe_to_use=safe,
        suspicious_claims=claims,
        summary=raw.get("summary", ""),
    )
