# LLM Output Eval Suite

**Tool 4 of 5 — llm-tools suite by [Swapnanil Saha](https://swapnanilsaha.com)**

A production-grade Python CLI + REST API that evaluates and compares LLM outputs with structured, multi-dimensional scoring. Give it a prompt and one or more LLM responses — from different models, different prompt variants, or different runs — and receive a rigorous evaluation: per-dimension scores, pairwise comparisons, winner declaration with reasoning, and actionable guidance on why one response outperforms another.

---

## The Business Problem

Teams building LLM-powered products have no systematic way to compare outputs. "Which prompt is better?" and "which model should we use?" are answered by gut feel and cherry-picked examples. This tool applies structured evaluation criteria — faithfulness, instruction-following, coherence, completeness, conciseness, and task-specific dimensions — to make LLM comparison **rigorous, repeatable, and auditable**. It brings the discipline of software testing to prompt engineering.

---

## Supported Task Types & Dimensions

| Task Type | Task-Specific Dimensions |
|-----------|--------------------------|
| `summarisation` | faithfulness (25%), coverage (20%), compression_quality (10%) |
| `qa` | faithfulness (30%), accuracy (25%), appropriate_hedging (0%) |
| `instruction_following` | format_compliance (25%), completeness (25%), constraint_adherence (10%) |
| `code_generation` | correctness (35%), readability (20%), best_practices (10%) |
| `creative_writing` | originality (25%), engagement (25%), tone_match (10%) |
| `classification` | accuracy (35%) |
| `extraction` | faithfulness (30%), completeness (25%) |
| `translation` | faithfulness (30%), accuracy (20%) |
| `general` | — |

**Universal dimensions** applied to every task type: instruction_following (20%), coherence (15%), conciseness (10%).

---

## Quick Start with Docker

```bash
# 1. Clone and set up environment
git clone <repo>
cd llm-eval-suite
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 2. Start the API server
docker-compose up api

# 3. Test the health endpoint
curl http://localhost:8000/health
# {"status":"ok","model":"claude-sonnet-4-6"}

# 4. Run a CLI evaluation
docker-compose run --rm --profile cli cli eval \
  --file examples/eval_qa.json \
  --mode compare \
  --format markdown
```

---

## CLI Usage

```bash
# Score a single response
python main.py eval --file examples/eval_summarisation.json --mode score

# Compare two responses (pairwise)
python main.py eval --file examples/eval_qa.json --mode compare

# Rank multiple responses (best to worst)
python main.py eval --file examples/eval_codegen.json --mode rank

# Quick inline evaluation
python main.py eval \
  --prompt "Summarise the key risks of AI in healthcare" \
  --response "AI in healthcare poses risks including..." \
  --task-type summarisation \
  --mode score

# Override dimensions
python main.py eval --file eval.json --dimensions faithfulness,coherence,conciseness

# Output formats
python main.py eval --file eval.json --format markdown
python main.py eval --file eval.json --format json
python main.py eval --file eval.json --format html --output report.html
python main.py eval --file eval.json --format junit --output results.xml

# Batch eval — evaluate multiple prompts from a directory
python main.py eval --batch examples/ --format markdown --output batch_report.md
```

---

## API Usage

### Start the server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# or: python api.py
```

### POST /eval — Full evaluation

```bash
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d @examples/eval_qa.json
```

### POST /eval/quick — Quick evaluation with defaults

```bash
curl -X POST http://localhost:8000/eval/quick \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "task_type": "qa",
    "responses": [
      {"label": "Model A", "response": "The capital of France is Paris."},
      {"label": "Model B", "response": "France capital is Paris, its a big city."}
    ]
  }'
```

### POST /eval/batch — Batch evaluation

```bash
curl -X POST http://localhost:8000/eval/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"prompt": "Summarise X", "task_type": "summarisation", "responses": [...], "eval_mode": "rank"},
    {"prompt": "Summarise Y", "task_type": "summarisation", "responses": [...], "eval_mode": "rank"}
  ]'
```

### GET /dimensions — Inspect all dimensions and weights

```bash
curl http://localhost:8000/dimensions
```

---

## Sample Eval: Hallucination Caught in Action

The `examples/eval_qa.json` file demonstrates a faithfulness evaluation. The prompt asks about a product's digital refund policy. Two responses are evaluated:

- **Response A** correctly states the 14-day policy as written in the FAQ document
- **Response B** confidently states "30-day policy" — a hallucination that confuses the physical product return window with the digital policy

Running this eval:

```bash
python main.py eval --file examples/eval_qa.json --mode compare --format markdown
```

Expected output highlights:
- Response A: faithfulness score **10/10** — every claim sourced from the FAQ
- Response B: faithfulness score **≤ 2/10** — "States 30-day policy but source says 14 days — clear hallucination"
- Winner: **Response A** by **clear** margin
- Key quote flagged: *"PixelPress offers a 30-day refund policy on digital products"*

This is exactly the kind of systematic quality gate that prevents hallucinating LLMs from reaching production.

---

## CI/CD Integration

The JUnit XML output format integrates with GitHub Actions, Jenkins, and any CI pipeline:

```yaml
# .github/workflows/llm-eval.yml
name: LLM Eval Regression Suite

on:
  pull_request:
  push:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run LLM eval
        run: |
          docker-compose run --rm --profile cli cli eval \
            --file evals/regression_suite.json \
            --mode rank \
            --format junit \
            --output results/eval_results.xml
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          
      - name: Publish eval results
        uses: mikepenz/action-junit-report@v3
        with:
          report_paths: results/eval_results.xml
```

**JUnit format:** Each evaluation dimension = one test case. A score below `EVAL_PASS_THRESHOLD` (default: 7.0) = test failure. Configure the threshold in `.env`:

```
EVAL_PASS_THRESHOLD=7.0
```

---

## Input Schema

```python
class EvalInput:
    prompt: str                    # original prompt sent to the LLM(s)
    task_type: str                 # one of the supported task types
    reference_answer: str | None   # ground truth if available
    source_document: str | None    # source doc for faithfulness eval
    responses: list[LLMResponse]   # 1 to 10 responses
    dimensions: list[str] | None   # override default dimensions
    weights: dict[str, float] | None  # override dimension weights
    eval_mode: "score" | "compare" | "rank"
    audience: str | None
    success_criteria: str | None

class LLMResponse:
    label: str         # e.g. "GPT-4o", "Claude Sonnet", "Prompt v2"
    response: str      # the raw LLM output to evaluate
    model: str | None
    latency_ms: int | None
    cost_usd: float | None
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests use mocked judge responses — no API key needed for the test suite.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Your Anthropic API key |
| `MODEL` | `claude-sonnet-4-6` | Judge model |
| `MAX_TOKENS` | `3000` | Max tokens for judge responses |
| `EVAL_PASS_THRESHOLD` | `7.0` | Minimum score for JUnit pass |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Project Structure

```
llm-eval-suite/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── main.py                      # CLI entry point (Typer)
├── api.py                       # FastAPI app
├── agent/
│   ├── evaluator.py             # Core evaluation logic + Anthropic SDK calls
│   ├── comparator.py            # Pairwise + multi-response comparison
│   ├── prompts.py               # Judge system prompts, dimension configs
│   └── models.py                # Pydantic input/output models
├── examples/
│   ├── eval_summarisation.json  # Summarisation task comparison (rank mode)
│   ├── eval_qa.json             # Q&A faithfulness evaluation (hallucination demo)
│   ├── eval_codegen.json        # Code generation quality (rank mode)
│   └── sample_output.json       # Sample EvalReport output
└── tests/
    ├── test_evaluator.py
    ├── test_comparator.py
    └── test_api.py
```

---

Built by [Swapnanil Saha](https://swapnanilsaha.com)
