from __future__ import annotations

JUDGE_SYSTEM_PROMPT = """You are an expert LLM evaluator. Your job is to assess LLM-generated responses with the precision of a senior AI researcher and the practicality of an engineer shipping a production system.

Evaluation principles:
- EVIDENCE-BASED: Every score must be justified with specific evidence from the response. "This is well-written" is not evidence. "The response correctly identifies all three causes mentioned in the source document without adding unsupported claims" is evidence.
- CALIBRATED: A score of 10 means the response cannot be meaningfully improved on this dimension. A score of 5 means it partially meets the criterion. Be willing to give 1s and 10s when warranted.
- INDEPENDENT: Evaluate each dimension independently. A response can be highly faithful but poorly structured. Score them separately.
- COMPARATIVE HONESTY: When comparing responses, declare a winner clearly. "Both are good" is not useful. Find the meaningful differences.
- QUOTE PRECISELY: When your score is driven by specific text in the response, quote it exactly in the quote field.

For faithfulness specifically:
- Any claim in the response not supported by the source document = hallucination. Flag every one. Do not give partial credit for mostly faithful responses if there are clear hallucinations.

Respond ONLY with valid JSON matching the output schema. No preamble, no markdown fences."""


DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "instruction_following": "Did the response do what was asked? Evaluate whether all explicit and implicit instructions in the prompt were followed completely.",
    "coherence": "Is the response logically structured? Evaluate flow, clarity, and whether ideas connect naturally.",
    "conciseness": "Is the response appropriately brief without losing substance? Penalise padding, repetition, or unnecessary elaboration.",
    "faithfulness": "Are all claims in the response supported by the source document or reference? Any unsupported claim is a hallucination — flag every one.",
    "coverage": "Did the response capture all key points from the source? Evaluate how much important content was included vs omitted.",
    "compression_quality": "Is the information density appropriate? Evaluate whether the response achieves a good ratio of information per word.",
    "accuracy": "Is the answer factually correct? Evaluate against the reference answer or ground truth.",
    "appropriate_hedging": "Does the response appropriately express uncertainty where warranted? Penalise overclaiming certainty on unclear or unsupported points.",
    "format_compliance": "Does the response follow the requested format, structure, and length constraints exactly?",
    "completeness": "Were all parts of the instruction addressed? A response that misses any sub-task is penalised proportionally.",
    "constraint_adherence": "Did the response respect any stated constraints (word limits, forbidden content, required inclusions)?",
    "correctness": "Is the code functionally correct? It should produce the right output for valid inputs and handle edge cases.",
    "readability": "Is the code clean and easy to understand? Evaluate naming, structure, spacing, and inline comments.",
    "best_practices": "Does the code use idiomatic patterns for the language? Evaluate for anti-patterns, security issues, and style.",
    "originality": "Is the response creative and non-generic? Penalise formulaic, predictable, or template-like writing.",
    "engagement": "Is the writing compelling and interesting to read? Evaluate narrative pull, voice, and reader experience.",
    "tone_match": "Does the tone match what was requested in the prompt? Evaluate formality, register, and emotional tone.",
}


TASK_DIMENSIONS: dict[str, dict[str, float]] = {
    "summarisation": {
        "instruction_following": 0.20,
        "coherence": 0.15,
        "conciseness": 0.10,
        "faithfulness": 0.25,
        "coverage": 0.20,
        "compression_quality": 0.10,
    },
    "qa": {
        "instruction_following": 0.20,
        "coherence": 0.15,
        "conciseness": 0.10,
        "faithfulness": 0.30,
        "accuracy": 0.25,
        "appropriate_hedging": 0.00,
    },
    "instruction_following": {
        "instruction_following": 0.20,
        "coherence": 0.15,
        "conciseness": 0.10,
        "format_compliance": 0.25,
        "completeness": 0.25,
        "constraint_adherence": 0.10,
    },
    "code_generation": {
        "instruction_following": 0.20,
        "coherence": 0.15,
        "conciseness": 0.10,
        "correctness": 0.35,
        "readability": 0.20,
        "best_practices": 0.10,
    },
    "creative_writing": {
        "instruction_following": 0.20,
        "coherence": 0.15,
        "conciseness": 0.10,
        "originality": 0.25,
        "engagement": 0.25,
        "tone_match": 0.10,
    },
    "classification": {
        "instruction_following": 0.30,
        "coherence": 0.20,
        "conciseness": 0.15,
        "accuracy": 0.35,
    },
    "extraction": {
        "instruction_following": 0.20,
        "coherence": 0.15,
        "conciseness": 0.10,
        "faithfulness": 0.30,
        "completeness": 0.25,
    },
    "translation": {
        "instruction_following": 0.20,
        "coherence": 0.20,
        "conciseness": 0.10,
        "faithfulness": 0.30,
        "accuracy": 0.20,
    },
    "general": {
        "instruction_following": 0.35,
        "coherence": 0.35,
        "conciseness": 0.30,
    },
}


def get_dimensions_for_task(task_type: str) -> dict[str, float]:
    return TASK_DIMENSIONS.get(task_type, TASK_DIMENSIONS["general"])


def build_scoring_prompt(
    prompt: str,
    response_label: str,
    response_text: str,
    dimensions: dict[str, float],
    task_type: str,
    reference_answer: str | None = None,
    source_document: str | None = None,
    audience: str | None = None,
    success_criteria: str | None = None,
    model_name: str | None = None,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
) -> str:
    parts = []

    parts.append(f"TASK TYPE: {task_type}")
    parts.append(f"ORIGINAL PROMPT:\n{prompt}")

    if source_document:
        parts.append(f"SOURCE DOCUMENT:\n{source_document}")
    if reference_answer:
        parts.append(f"REFERENCE ANSWER:\n{reference_answer}")
    if audience:
        parts.append(f"INTENDED AUDIENCE: {audience}")
    if success_criteria:
        parts.append(f"SUCCESS CRITERIA: {success_criteria}")

    parts.append(f"\nRESPONSE TO EVALUATE (label: {response_label}):\n{response_text}")

    if model_name:
        parts.append(f"Model: {model_name}")
    if latency_ms is not None:
        parts.append(f"Latency: {latency_ms}ms")
    if cost_usd is not None:
        parts.append(f"Cost: ${cost_usd:.6f}")

    dim_lines = []
    for dim, weight in dimensions.items():
        desc = DIMENSION_DESCRIPTIONS.get(dim, f"Evaluate {dim}.")
        dim_lines.append(f"  - {dim} (weight {weight:.0%}): {desc}")

    parts.append("\nEVALUATE ON THESE DIMENSIONS:\n" + "\n".join(dim_lines))

    schema = {
        "label": response_label,
        "overall_score": "<weighted composite 1.0-10.0>",
        "dimensions": [
            {
                "dimension": "<name>",
                "score": "<1.0-10.0>",
                "reasoning": "<specific evidence-based explanation>",
                "quote": "<exact quote from response that drove the score, or null>",
            }
        ],
        "strengths": ["<specific strength 1>", "<specific strength 2>"],
        "weaknesses": ["<specific weakness 1>"],
        "one_line_verdict": "<one sentence summary of this response's quality>",
    }

    import json
    parts.append(f"\nRespond with JSON matching this exact schema:\n{json.dumps(schema, indent=2)}")

    return "\n\n".join(parts)


def build_comparison_prompt(
    prompt: str,
    response_a_label: str,
    response_a_text: str,
    response_b_label: str,
    response_b_text: str,
    task_type: str,
    reference_answer: str | None = None,
    source_document: str | None = None,
) -> str:
    parts = []

    parts.append(f"TASK TYPE: {task_type}")
    parts.append(f"ORIGINAL PROMPT:\n{prompt}")

    if source_document:
        parts.append(f"SOURCE DOCUMENT:\n{source_document}")
    if reference_answer:
        parts.append(f"REFERENCE ANSWER:\n{reference_answer}")

    parts.append(f"RESPONSE A ({response_a_label}):\n{response_a_text}")
    parts.append(f"RESPONSE B ({response_b_label}):\n{response_b_text}")

    parts.append(
        "Compare these two responses directly. Identify the most meaningful differences. "
        "Declare a clear winner (or 'tie' only if truly indistinguishable). "
        "Assess margin as 'clear' (significant quality gap), 'slight' (minor edge), or 'tie'."
    )

    schema = {
        "response_a": response_a_label,
        "response_b": response_b_label,
        "winner": f"<{response_a_label} | {response_b_label} | tie>",
        "margin": "<clear | slight | tie>",
        "reasoning": "<specific evidence-based explanation of why winner beats loser>",
    }

    import json
    parts.append(f"\nRespond with JSON matching this exact schema:\n{json.dumps(schema, indent=2)}")

    return "\n\n".join(parts)


def build_ranking_prompt(
    prompt: str,
    responses: list[dict],
    task_type: str,
    reference_answer: str | None = None,
    source_document: str | None = None,
) -> str:
    parts = []

    parts.append(f"TASK TYPE: {task_type}")
    parts.append(f"ORIGINAL PROMPT:\n{prompt}")

    if source_document:
        parts.append(f"SOURCE DOCUMENT:\n{source_document}")
    if reference_answer:
        parts.append(f"REFERENCE ANSWER:\n{reference_answer}")

    for r in responses:
        parts.append(f"RESPONSE ({r['label']}):\n{r['response']}")

    labels = [r["label"] for r in responses]
    parts.append(
        "Rank all responses from best to worst. Declare an overall winner. "
        "Provide reasoning explaining why the winner outperforms the rest. "
        "Be specific about what makes the top response better."
    )

    schema = {
        "ranking": labels,
        "winner": labels[0],
        "winner_reasoning": "<specific explanation of why the winner beats the rest>",
        "key_insight": "<single most important finding from this comparison>",
        "recommendation": "<what to do next based on these results>",
    }

    import json
    parts.append(f"\nRespond with JSON matching this exact schema (reorder ranking as you see fit):\n{json.dumps(schema, indent=2)}")

    return "\n\n".join(parts)
