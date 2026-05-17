# llm-eval-suite
> Stop comparing LLM outputs by gut feel. Score them across task-specific dimensions — with evidence, not opinion.

Part of the [llm-tools suite](https://github.com/swapnanil) by [Swapnanil Saha](https://swapnanilsaha.com)

## What it does

Teams switch models, upgrade prompts, and change fine-tuning datasets — then decide which is better by reading a few outputs and going with gut feel. LLM Eval Suite scores every response across task-specific dimensions with evidence. Plug JUnit XML output into GitHub Actions and model upgrades either pass or fail like any other test.

## Quick start

```bash
git clone https://github.com/swapnanil/llm-eval-suite
cd llm-eval-suite
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker-compose up api
```

## CLI usage

```bash
# Compare two responses
docker-compose run cli eval \
  --file examples/eval_qa.json \
  --mode compare --format markdown

# Rank multiple responses
docker-compose run cli eval \
  --file examples/eval_summarisation.json \
  --mode rank --format json

# CI/CD — output JUnit XML for GitHub Actions
docker-compose run cli eval \
  --file evals/suite.json \
  --mode rank --format junit --output results.xml
```

## API usage

```bash
# Evaluate responses
curl -X POST http://localhost:8000/eval \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the refund policy?", "task_type": "qa", "source_document": "Digital products are eligible for a full refund within 14 days if unused.", "responses": [{"label": "A", "response": "Refund within 14 days if unused."}, {"label": "B", "response": "30-day return policy, no questions asked."}], "eval_mode": "compare"}'
```

## Input / Output

**Input:**
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

**Output excerpt:**
```json
{
  "winner": "Response A",
  "margin": "clear",
  "scores": {
    "Response B": {
      "faithfulness": {
        "score": 1.0,
        "reasoning": "States '30-day policy' — source specifies 14 days. Clear hallucination.",
        "quote": "30-day return policy, no questions asked"
      }
    }
  }
}
```

## GitHub Actions integration

```yaml
- name: Run LLM eval
  run: docker-compose run cli eval --file evals/suite.json --mode rank --format junit --output results.xml
- uses: mikepenz/action-junit-report@v3
  with:
    report_paths: results.xml
```

## Built with

- Python 3.11
- Anthropic SDK (claude-sonnet-4-6)
- FastAPI + uvicorn
- Docker + docker-compose
- pytest

## Author

Swapnanil Saha · [swapnanilsaha.com](https://swapnanilsaha.com) · [LinkedIn](https://linkedin.com/in/swapnanil)
