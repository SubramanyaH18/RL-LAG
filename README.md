# RL-LAG Architectural Prototype — Ollama Edition

This repository implements the RL-LAG prototype: multi-hop QA with RL-guided DAG
reasoning over HotpotQA.  The LLM (qwen2.5:3b-instruct via Ollama) is frozen
throughout; three small MLP policy networks (pi^G, pi^R, pi^C) are trained with PPO.

## Architecture

1. `decomposition.py` — converts a question into at most six atomic subproblems.
2. `graph_builder.py` — constructs and validates a NetworkX DAG.
3. `retrieval.py`     — local FAISS retriever (all-MiniLM-L6-v2) with context-aware
                        query enrichment (A1), type-aware k (A3), and cosine dedup (A5).
4. `solver.py`        — resolves nodes in topological order through Ollama.
5. `reward.py`        — heuristic reward (correctness, retrieval presence, token efficiency,
                        logical consistency, grounding).
6. `policies.py`      — three MLP actor-critic networks + checkpoint I/O.
7. `train_ppo.py`     — PPO training loop with epoch-based question sampling.
8. `data_pools.py`    — builds and caches the fixed train + eval question pools.
9. `sanity_check.py`  — pre-flight component validator (run before training).
10. `eval.py`         — 4-way EM/F1 evaluation with McNemar + bootstrap statistics.
11. `app.py`          — Streamlit UI for interactive demos.

## Prerequisites

- Python 3.10+
- Ollama installed and running with `qwen2.5:3b-instruct`

```bash
ollama pull qwen2.5:3b-instruct
```

## Corpus and Question Pools

| File | Questions | Purpose |
|------|-----------|---------|
| `corpus/train_pool.json`        | **2000** (fixed seed 42) | PPO training — epoch-sampled |
| `corpus/eval_pool.json`         | **300** (zero overlap)   | All reported EM/F1 results  |
| `corpus/hotpot_questions.jsonl` | 25                       | Smoke-test / Streamlit demo only |

All questions are `hard`/`medium` `bridge`+`comparison` from HotpotQA validation/distractor.

> **Pool size derivation:** Measured dry-run (PPO + FAISS) = 0.82 s/ep; real local (+ Ollama CPU)
> = 3.1 s/ep mean; Kaggle T4 estimate = 1.6 s/ep (3× GPU speedup). T4 session ceiling ≈ 26,000
> episodes. `pool=2000` → **13 epochs/session** — squarely in the 5–15 epoch sweet spot.

### One-time pool build

```powershell
pip install datasets
python data_pools.py          # downloads HotpotQA once, writes both pool files
```

## Setup

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python data_pools.py                  # one-time: build train + eval pools
python scripts/build_hotpot_subset.py # one-time: build 25-q demo corpus for Streamlit
streamlit run app.py
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python data_pools.py
python scripts/build_hotpot_subset.py
streamlit run app.py
```

## PPO Training

```powershell
# Pre-flight sanity check (recommended before first run)
python sanity_check.py

# Full T4 run — 26,000 steps, checkpoint every 500
python train_ppo.py --steps 26000 --checkpoint-every 500 --device cuda

# Resume (skip sanity check to save ~2 min on resume)
python train_ppo.py --steps 26000 --checkpoint-every 500 --device cuda --skip-sanity

# Dry-run (no Ollama needed)
python train_ppo.py --dry-run --steps 5 --skip-sanity

# Local CPU (slower, for testing only)
python train_ppo.py --steps 500 --checkpoint-every 100 --skip-sanity
```

Sampling is **epoch-based**: the 2000-question pool is shuffled and exhausted before
reshuffling. One Kaggle T4 session (12h) ≈ **13 epochs** (~26,000 episodes).
The current epoch is shown in the step log: `[step 04200|ep3]`.

## Evaluation

```powershell
# Full 4-way eval on the 300-question held-out eval pool (for reported results)
python eval.py

# Baseline only (no Ollama needed)
python eval.py --eval-mode baseline-only

# Fast smoke-test on the 25-question demo subset (not for reported results)
python eval.py --smoke-test --eval-mode baseline-only

# Skip statistical tests (faster)
python eval.py --no-stats
```

`results.json` contains:
- `summary`     — EM/F1 per condition
- `statistics`  — McNemar p-values and bootstrap 95% CI on F1 (PPO vs each baseline)
- `per_question` — per-question predictions
- `metadata`    — pool description and training steps

## Sanity Check

```powershell
python sanity_check.py            # first run records baseline to sanity_baseline.json
python sanity_check.py            # subsequent runs compare vs baseline, warn on drops
python sanity_check.py --dry-run  # mock LLM, no Ollama needed
python sanity_check.py --reset-baseline  # re-record after intentional code changes
```

Four checks are run on 50 training-pool questions:
- **A** Decomposition validity rate (valid DAG, no cycles, non-trivial)
- **B** Retrieval A1 context-enrichment effect (% nodes where top-5 set changes)
- **C** Dedup A5 activity (mean passages removed by 0.92 cosine threshold)
- **D** Type-aware k A3 behaviour (temporal/comparative k bounds)

## Test

```bash
pytest -q
```

## Notes

- The FAISS index is rebuilt only when the corpus or embedding model changes.
- Checkpoint files include the EpochSampler state, so training resumes with correct
  epoch position even after a session break.
- All reported numbers come from `eval_pool.json` (300 held-out questions).
  The 25-question demo file is used only for Streamlit and smoke-tests.
- Ollama has no request-rate limit, so Groq-specific quota logic has been replaced
  by transient-failure retries and local usage metrics.
