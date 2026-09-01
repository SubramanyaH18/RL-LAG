# RL-LAG Architectural Prototype — Ollama Edition

This repository implements the uploaded RL-LAG prototype specification while replacing Groq with a **local Ollama model**. It demonstrates the pipeline shape only: LLM decomposition, dependency DAG construction, local FAISS retrieval, sequential node resolution, answer synthesis, and a rule-based reward placeholder.

It does **not** train PPO, reproduce the paper's experimental results, fine-tune a model, or use a large corpus.

## Architecture

1. `decomposition.py` converts a question into at most six atomic subproblems.
2. `graph_builder.py` constructs and validates a NetworkX DAG.
3. `retrieval.py` embeds the small local corpus and searches it with FAISS.
4. `solver.py` resolves nodes in topological order through Ollama.
5. `reward.py` computes an explicitly non-learned heuristic reward.
6. `app.py` displays all stages in Streamlit.

All Ollama calls pass through `llm_client.py`, which adds response caching, exponential retry, token/performance logging, and session counters.

## Prerequisites

- Python 3.10 or newer
- Ollama installed and running
- A small local model pulled in Ollama

The default model is `qwen2.5:3b-instruct`, keeping the project within the original specification's prohibition on 7B/13B/70B models. (Swapped from the earlier `llama3.2:3b` default — same size class, stronger instruction-following on the structured JSON prompts this pipeline relies on for decomposition/synthesis.)

```bash
ollama pull qwen2.5:3b-instruct
```

## Corpus — HotpotQA Subset

The retrieval corpus is a **25-question, deduplicated-paragraph subset** of the
[HotpotQA](https://hotpotqa.github.io/) `validation` / `distractor` split.

> **Important:** This is NOT the full HotpotQA dataset (90 k+ training questions)
> and NOT the paper's 21 M-passage Wikipedia retrieval index.
> It is a small, fixed-seed sample intended for local architectural demonstration only.

- Split: `validation` (distractor config)
- Filter: `level ∈ {hard, medium}` AND `type ∈ {bridge, comparison}`
- Sample size: 25 questions (seed 42 — deterministic)
- Paragraphs: deduplicated across all 25 examples' context fields

### One-time corpus build

```powershell
pip install datasets          # already in requirements.txt
python scripts/build_hotpot_subset.py
```

This produces:
- `corpus/hotpot_corpus.jsonl` — paragraphs for FAISS indexing
- `corpus/hotpot_questions.jsonl` — questions for the Streamlit dropdown

---

## Setup

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/build_hotpot_subset.py   # one-time corpus build
streamlit run app.py
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/build_hotpot_subset.py   # one-time corpus build
streamlit run app.py
```

Ollama normally exposes its local API at `http://localhost:11434`. Change `.env` when your daemon or model differs.

## Demo mode

The sidebar toggle **Use cached demo run** replays pre-recorded HotpotQA runs from
`demo_cache.json` with zero Ollama calls. Disable it to use live local inference
(requires Ollama running with `qwen2.5:3b-instruct`).

## Test

```bash
pytest -q
```

## Notes

- The first live run downloads the `all-MiniLM-L6-v2` embedding model unless it is already cached.
- The FAISS index is rebuilt only when the corpus or embedding model changes.
- Ollama has no Groq free-tier request limit, so Groq-specific `Retry-After` and quota logic has been replaced by transient-failure retries and local usage metrics.
- Live answer quality depends on the local Ollama model. The included cached demonstrations are deterministic presentation fallbacks.
