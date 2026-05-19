from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.hallucination import detect_hallucinations
from agent.models import HallucinationReport, SuspiciousClaim


SOURCE = "The Eiffel Tower was completed in 1889 and stands 330 metres tall."

CLEAN_RESPONSE = "The Eiffel Tower was completed in 1889 and is 330 metres tall."

HALLUCINATED_RESPONSE = (
    "The Eiffel Tower was built in 1850 and stands 400 metres tall. "
    "It was designed by Gustave Moulin."
)


def _mock_judge(raw: dict):
    return patch("agent.hallucination._call_judge", return_value=raw)


def _make_claim(claim: str, severity: str = "moderate", is_supported: bool = False, evidence: str = "") -> dict:
    return {
        "claim": claim,
        "severity": severity,
        "is_supported": is_supported,
        "evidence_or_gap": evidence,
    }


class TestDetectHallucinations:
    def test_returns_hallucination_report(self):
        raw = {
            "risk_level": "none",
            "hallucination_rate": 0.0,
            "safe_to_use": True,
            "suspicious_claims": [],
            "summary": "No hallucinations detected.",
        }
        with _mock_judge(raw):
            result = detect_hallucinations(CLEAN_RESPONSE, SOURCE)
        assert isinstance(result, HallucinationReport)

    def test_clean_response_safe_to_use(self):
        raw = {
            "risk_level": "none",
            "hallucination_rate": 0.0,
            "safe_to_use": True,
            "suspicious_claims": [],
            "summary": "All claims supported.",
        }
        with _mock_judge(raw):
            result = detect_hallucinations(CLEAN_RESPONSE, SOURCE)
        assert result.safe_to_use is True
        assert result.risk_level == "none"
        assert result.hallucination_rate == 0.0

    def test_hallucinated_response_flagged(self):
        raw = {
            "risk_level": "critical",
            "hallucination_rate": 0.67,
            "safe_to_use": False,
            "suspicious_claims": [
                _make_claim("built in 1850", "critical"),
                _make_claim("400 metres tall", "critical"),
                _make_claim("Gustave Moulin", "moderate"),
            ],
            "summary": "Multiple critical hallucinations found.",
        }
        with _mock_judge(raw):
            result = detect_hallucinations(HALLUCINATED_RESPONSE, SOURCE)
        assert result.safe_to_use is False
        assert result.risk_level == "critical"
        assert len(result.suspicious_claims) == 3

    def test_suspicious_claims_parsed_correctly(self):
        raw = {
            "risk_level": "moderate",
            "hallucination_rate": 0.5,
            "safe_to_use": False,
            "suspicious_claims": [
                _make_claim("built in 1850", "critical", False, "Source says 1889"),
                _make_claim("stands 330 metres", "low", True, "Exact match in source"),
            ],
            "summary": "One critical hallucination.",
        }
        with _mock_judge(raw):
            result = detect_hallucinations(HALLUCINATED_RESPONSE, SOURCE)
        assert result.suspicious_claims[0].severity == "critical"
        assert result.suspicious_claims[0].is_supported is False
        assert result.suspicious_claims[1].is_supported is True

    def test_custom_label_preserved(self):
        raw = {
            "risk_level": "low",
            "hallucination_rate": 0.1,
            "safe_to_use": True,
            "suspicious_claims": [],
            "summary": "Mostly clean.",
        }
        with _mock_judge(raw):
            result = detect_hallucinations(CLEAN_RESPONSE, SOURCE, response_label="GPT-4")
        assert result.response_label == "GPT-4"

    def test_invalid_risk_level_defaults_to_moderate(self):
        raw = {
            "risk_level": "unknown_value",
            "hallucination_rate": 0.3,
            "safe_to_use": False,
            "suspicious_claims": [],
            "summary": "Unclear.",
        }
        with _mock_judge(raw):
            result = detect_hallucinations(CLEAN_RESPONSE, SOURCE)
        assert result.risk_level == "moderate"

    def test_invalid_severity_defaults_to_low(self):
        raw = {
            "risk_level": "low",
            "hallucination_rate": 0.1,
            "safe_to_use": True,
            "suspicious_claims": [
                {"claim": "some claim", "severity": "INVALID", "is_supported": False, "evidence_or_gap": ""}
            ],
            "summary": "One claim.",
        }
        with _mock_judge(raw):
            result = detect_hallucinations(CLEAN_RESPONSE, SOURCE)
        assert result.suspicious_claims[0].severity == "low"

    def test_hallucination_rate_clamped_to_0_1(self):
        raw = {
            "risk_level": "high",
            "hallucination_rate": 5.0,
            "safe_to_use": False,
            "suspicious_claims": [],
            "summary": "Extremely high rate.",
        }
        with _mock_judge(raw):
            result = detect_hallucinations(HALLUCINATED_RESPONSE, SOURCE)
        assert result.hallucination_rate == 1.0

    def test_empty_response_raises_value_error(self):
        with pytest.raises(ValueError, match="Response text cannot be empty"):
            detect_hallucinations("   ", SOURCE)

    def test_empty_source_raises_value_error(self):
        with pytest.raises(ValueError, match="Source document cannot be empty"):
            detect_hallucinations(CLEAN_RESPONSE, "   ")

    def test_system_prompt_restored_after_call(self):
        import agent.prompts as _prompts
        original = _prompts.JUDGE_SYSTEM_PROMPT
        raw = {
            "risk_level": "none",
            "hallucination_rate": 0.0,
            "safe_to_use": True,
            "suspicious_claims": [],
            "summary": "Clean.",
        }
        with _mock_judge(raw):
            detect_hallucinations(CLEAN_RESPONSE, SOURCE)
        assert _prompts.JUDGE_SYSTEM_PROMPT is original
