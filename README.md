# Thesis Experiment: Prompt Injection in a RAG Chatbot

This repository contains a controlled experiment for a bachelor's thesis on **direct prompt injection** against a **RAG customer-support chatbot** and the effectiveness of mitigations **M1–M6** across these model backends:

- **OpenAI (API)**: `gpt-4o-mini`
- **Google (API)**: `gemini-2.5-flash-lite` (via `--llm google`)
- **Ollama (local)**: `llama3.1:8b`, `mistral`, `qwen3:8b`, `gemma2:9b`

The repo intentionally includes:

- An **attack suite** (`attacks/attack_suite_alt_fixed.json`)
- **Experiment logs** (`experiment/logs/*.jsonl`)

## Quickstart

From `thesis-chatbot/`:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create a `.env` file:

- Copy `.env.example` to `.env`
- Fill in `OPENAI_API_KEY` (required for embeddings / M6 guardrail), and `GOOGLE_API_KEY` if you use `--llm google`

## Run the experiment

Dry run (no API calls; validates the loop and logging):

```bash
python -m experiment.runner --llm openai --dry-run --suite alt_fixed
```

Small smoke test (first 10 prompts, 1 run, two mitigations):

```bash
python -m experiment.runner --llm openai --suite alt_fixed --max-prompts 10 --runs 1 --configs M1 M2
```

Full runs (defaults: all configs):

```bash
python -m experiment.runner --llm openai --suite alt_fixed --runs 1
python -m experiment.runner --llm google --suite alt_fixed --runs 1
python -m experiment.runner --llm ollama --suite alt_fixed --runs 1 --model llama3.1:8b
python -m experiment.runner --llm ollama --suite alt_fixed --runs 1 --model mistral
python -m experiment.runner --llm ollama --suite alt_fixed --runs 1 --model qwen3:8b
python -m experiment.runner --llm ollama --suite alt_fixed --runs 1 --model gemma2:9b
```

Logs are written to `experiment/logs/run_<provider>_<model>_<suite>_<timestamp>.jsonl`.

## Analysis

```bash
python -m analysis.analyze_results --logs experiment/logs
python -m analysis.analyze_results --logs experiment/logs --json > metrics.json
```

## Notes (reproducibility)

- RAG uses **Chroma** + **OpenAI embeddings** (`text-embedding-3-small`) as implemented in `chatbot/rag_chain.py`.
  - This means **`OPENAI_API_KEY` is required** for non-dry-run runs, even when `--llm ollama` is used for the chat model.
- **M6 guardrail** uses an additional OpenAI model (default `gpt-4o`). Override via `GUARDRAIL_OPENAI_MODEL`.

## Repository structure

- `attacks/` – prompt injection suite used by the runner
- `knowledge_base/` – text knowledge base used for retrieval (includes **fake** sensitive strings for exfiltration detection)
- `chatbot/` – RAG chain, model factory, mitigations, detection utilities
- `experiment/runner.py` – orchestrates suite × configs × repeats and writes JSONL logs
- `experiment/logs/` – collected JSONL runs
- `analysis/` – metrics and reporting scripts
- `docs/` – implementation details used during write-up
