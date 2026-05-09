# Thesis Experiment: Prompt Injection in a RAG Chatbot

Controlled experiment for **direct prompt injection** against a **RAG customer-support chatbot** and mitigation effectiveness **M1–M6** across these backends:

- **OpenAI (API)**: `gpt-4o-mini`
- **Google (API)**: `gemini-2.5-flash-lite` (`--llm google`)
- **Ollama (local)**: `llama3.1:8b`, `mistral`, `qwen3:8b`, `gemma2:9b`

The repo ships:

- **Attack suite:** `attacks/attack_suite_grok4_elicit_half.json` (pass `--suite attack_suite_grok4_elicit_half.json`)
- **Benign prompts (utility):** `attacks/benign_prompt_suite.json`

Run outputs (`experiment/logs/*.jsonl`, `experiment/metrics_out/`) are **local only** — they are gitignored; regenerate them with the runner and analysis scripts below.

## Quickstart

From `thesis-chatbot/`:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create a `.env` file:

- Copy `.env.example` to `.env`
- **`OPENAI_API_KEY`** — required only when using `--llm openai`
- **`GOOGLE_API_KEY`** — required when using `--llm google`
- **RAG embeddings** use **Ollama** locally (see below); no OpenAI embedding key needed for retrieval
- **M6** uses a **local Ollama** classifier model (see below)

Ensure **Ollama** is running with at least:

- An embedding model (default **`nomic-embed-text`**) — override with **`EMBEDDINGS_MODEL`**
- For M6 runs: **`granite3.2:8b`** (or whatever you set in **`GUARDRAIL_MODEL`**)

## Run the experiment

Pass **`--suite attack_suite_grok4_elicit_half.json`** so the runner loads `attacks/attack_suite_grok4_elicit_half.json`.

Dry run (validates loop and logging without full experiment cost):

```bash
python -m experiment.runner --llm openai --dry-run --suite attack_suite_grok4_elicit_half.json
```

Small smoke test (first 10 prompts, 1 run, two mitigations):

```bash
python -m experiment.runner --llm openai --suite attack_suite_grok4_elicit_half.json --max-prompts 10 --runs 1 --configs M1 M2
```

Full runs (defaults: all mitigation configs):

```bash
python -m experiment.runner --llm openai --suite attack_suite_grok4_elicit_half.json --runs 1
python -m experiment.runner --llm google --suite attack_suite_grok4_elicit_half.json --runs 1
python -m experiment.runner --llm ollama --suite attack_suite_grok4_elicit_half.json --runs 1 --model llama3.1:8b
python -m experiment.runner --llm ollama --suite attack_suite_grok4_elicit_half.json --runs 1 --model mistral
python -m experiment.runner --llm ollama --suite attack_suite_grok4_elicit_half.json --runs 1 --model qwen3:8b
python -m experiment.runner --llm ollama --suite attack_suite_grok4_elicit_half.json --runs 1 --model gemma2:9b
```

Logs are written to `experiment/logs/run_<provider>_<model>_<timestamp>.jsonl` (suite name is not repeated in the filename; keep separate runs in separate files).

## Analysis

Aggregate metrics from JSONL logs:

```bash
python -m analysis.analyze_results --logs experiment/logs
python -m analysis.analyze_results --logs experiment/logs --json > metrics.json
```

Thesis-style Markdown tables (same layout as prior thesis repro tables):

```bash
python experiment/compute_thesis_tables.py --input-dir path/to/jsonl_logs --out experiment/metrics_out/thesis_repro/tables.md
```

## Notes (reproducibility)

- **RAG / retrieval** — **`chatbot/rag_chain.py`** uses **Chroma** with **Ollama embeddings** (`langchain_ollama.OllamaEmbeddings`). Default model: **`nomic-embed-text`**. Override with **`EMBEDDINGS_MODEL`** in `.env`.
  - Fully **local** retrieval is possible: **`--llm ollama`** + Ollama embeddings + local chat model.
  - OpenAI is **not** used for embeddings in the current code path.
- **M6 (guardrail LLM)** — **`chatbot/mitigations/m6_guardrail_llm.py`**: a **local Ollama** chat model classifies each user turn as **SAFE** vs **INJECTION** (fixed system prompt). Default model: **`granite3.2:8b`**. Override with **`GUARDRAIL_MODEL`**. Set **`M6_DISABLE=1`** to bypass the guardrail for debugging.

## Repository structure

- `attacks/` — attack (and benign) prompt suites used by the runner
- `knowledge_base/` — text KB for retrieval (includes **fake** sensitive strings for exfiltration checks)
- `chatbot/` — RAG chain, LLM factory, mitigations, detection helpers
- `experiment/runner.py` — suite × configs × repeats → JSONL logs
- `experiment/compute_thesis_tables.py` — builds thesis Markdown tables from JSONL
- `experiment/logs/`, `experiment/metrics_out/` — generated locally (ignored by Git)
- `analysis/` — log parsing, metrics CLI (also imported by `compute_thesis_tables.py`)

Implementation notes for the write-up may live under `docs/` locally; that folder is gitignored in this project.
