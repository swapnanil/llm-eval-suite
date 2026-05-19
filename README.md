# llm-eval-suite
> Stop comparing LLM outputs by gut feel. Score them across task-specific dimensions — with evidence, not opinion.

Part of the [llm-tools suite](https://github.com/swapnanil) by [Swapnanil Saha](https://swapnanilsaha.com)

## What it does

Teams switch models, upgrade prompts, and change fine-tuning datasets — then decide which is better by reading a few outputs and going with gut feel. LLM Eval Suite scores every response across task-specific dimensions with evidence, runs regression checks against saved baselines, detects hallucinations claim-by-claim, quantifies how sensitive your scores are to prompt wording, and eliminates judge bias through multi-judge panel consensus.

## Features

| Feature | Description |
|---|---|
| Multi-dimensional scoring | Task-specific rubrics for 10 task types (QA, summarisation, RAG, code, etc.) |
| Pairwise ranking | Head-to-head or full ranking across N responses |
| Regression testing | Save baselines, run comparisons, exit non-zero in CI when scores drop |
| Hallucination detection | Claim-level analysis against a source document with risk levels |
| Prompt sensitivity analysis | Variance per dimension across prompt variants — know which dimensions are fragile |
| Panel evaluation | N independent judge passes → consensus score + disagreement flags |
| JUnit XML output | Plug directly into GitHub Actions or any CI runner |
| RAGAS-compatible | `rag` task type maps faithfulness, answer relevancy, context precision, context recall |

## Quick start

```bash
git clone https://github.com/swapnanil/llm-eval-suite
cd llm-eval-suite
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker-compose up api
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/eval` | Score or compare responses |
| POST | `/eval/quick` | Minimal payload eval |
| POST | `/eval/batch` | Batch multiple evals in one call |
| POST | `/eval/panel` | Multi-judge consensus evaluation |
| POST | `/hallucination` | Claim-level hallucination detection |
| POST | `/sensitivity` | Prompt sensitivity analysis |
| POST | `/regression/baselines` | Save an eval report as a baseline |
| GET | `/regression/baselines` | List saved baselines |
| POST | `/regression/run` | Compare a report against a baseline |
| GET | `/dimensions` | Browse task types and their dimensions |

## CLI usage

```bash
# Compare two responses (markdown output)
docker-compose run cli eval \
  --file examples/eval_qa.json \
  --mode compare --format markdown

# Rank multiple responses (JUnit for CI)
docker-compose run cli eval \
  --file evals/suite.json \
  --mode rank --format junit --output results.xml

# Detect hallucinations in a response
docker-compose run cli hallucination \
  --response output.txt --source source.txt --format markdown

# Run regression check against a saved baseline
docker-compose run cli regression run \
  results.json --id prod-v1 --format markdown

# Analyse how sensitive scores are to prompt wording
docker-compose run cli sensitivity prompts.json response.txt \
  --task-type summarisation --format markdown

# Panel evaluation (3 independent judge passes)
docker-compose run cli eval panel eval_input.json --judges 3
```

## Regression testing in CI

```yaml
- name: Run LLM eval
  run: docker-compose run cli eval --file evals/suite.json --mode rank --format junit --output results.xml
- uses: mikepenz/action-junit-report@v3
  with:
    report_paths: results.xml

- name: Regression check
  run: docker-compose run cli regression run results.json --id prod-baseline
  # exits 1 if any dimension drops beyond threshold
```

## Input / Output

**Eval input:**
```json
{
  "prompt": "What is the refund policy for digital products?",
  "task_type": "qa",
  "source_document": "Digital products are eligible for a full refund within 14 days if unused.",
  "responses": [
    {"label": "Response A", "response": "Refund within 14 days if unused."},
    {"label": "Response B", "response": "30-day return policy, no questions asked."}
  ],
  "eval_mode": "compare"
}
```

**Hallucination request:**
```json
{
  "response_text": "The Eiffel Tower was built in 1850 and stands 400 metres tall.",
  "source": "The Eiffel Tower was completed in 1889 and stands 330 metres tall.",
  "response_label": "gpt-4o"
}
```

**Hallucination output excerpt:**
```json
{
  "risk_level": "critical",
  "hallucination_rate": 1.0,
  "safe_to_use": false,
  "suspicious_claims": [
    {"claim": "built in 1850", "severity": "critical", "is_supported": false, "evidence_or_gap": "Source says 1889"},
    {"claim": "400 metres tall", "severity": "critical", "is_supported": false, "evidence_or_gap": "Source says 330 metres"}
  ]
}
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `MODEL` | `claude-sonnet-4-6` | Judge model |
| `MAX_TOKENS` | `3000` | Max tokens per judge call |
| `BASELINE_DIR` | `./baselines` | Directory for saved regression baselines |
| `EVAL_PASS_THRESHOLD` | `7.0` | Score floor for JUnit pass/fail |
| `LOG_LEVEL` | `INFO` | Logging level |

## Built with

- Python 3.11
- Anthropic SDK (`claude-sonnet-4-6`)
- FastAPI + uvicorn
- Docker + docker-compose
- pytest (92 tests)

## Author

Swapnanil Saha · [swapnanilsaha.com](https://swapnanilsaha.com) · [LinkedIn](https://linkedin.com/in/swapnanil)
