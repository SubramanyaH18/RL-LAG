"""Fixed, cacheable training and evaluation pools for RL-LAG PPO training.

Replaces the previous dynamic 17k-question sampling regime with two
non-overlapping, seeded, persisted pools:

  corpus/train_pool.json  -- fixed pool of TRAIN_POOL_SIZE questions
                             sampled once, reused across all training runs.
  corpus/eval_pool.json   -- held-out pool of EVAL_POOL_SIZE questions
                             with zero overlap with train_pool (deduped by ID).
                             Used for all reported EM/F1 numbers.

The original corpus/hotpot_questions.jsonl (25-question demo subset) is
preserved and still usable as a fast smoke-test; it is NOT used for any
reported evaluation results.

Usage
-----
    # Build both pools (one-time, ~200 MB download if HotpotQA not cached)
    python data_pools.py

    # From code
    from data_pools import load_train_pool, load_eval_pool
    train_qs = load_train_pool()   # list[dict]: id, question, answer, type, level
    eval_qs  = load_eval_pool()
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
# Derived from measured timing (2026-09-03):
#   Dry-run warm:   0.82 s/ep  (PPO + FAISS, no LLM)
#   Local CPU+Ollama: 3.1 s/ep mean (600 eps observed)
#   Kaggle T4 est.: 1.6 s/ep  (3x GPU speedup on qwen2.5:3b)
#   T4 session ceiling: ~26,000 eps (12h - 30min setup)
#   pool=2000 -> 13.1 epochs/session  (target sweet spot: 5-15 epochs)
#   pool= 750 -> 34.9 epochs/session  (over-repeating, overfitting risk)
TRAIN_POOL_SIZE: int = 2000   # questions in the fixed training pool
EVAL_POOL_SIZE:  int = 300    # questions in the held-out evaluation pool
POOL_SEED:       int = 42     # fixed seed -- must not change once pools are built


ROOT            = Path(__file__).resolve().parent
TRAIN_POOL_PATH = ROOT / "corpus" / "train_pool.json"
EVAL_POOL_PATH  = ROOT / "corpus" / "eval_pool.json"
_SOURCE_JSONL   = ROOT / "corpus" / "hotpot_questions.jsonl"
_SOURCE_MIN     = 500   # fewer than this => treat as demo subset, download full


# ── Source loader ──────────────────────────────────────────────────────────────

def _load_source() -> list[dict]:
    """Load filtered source questions. Downloads from HuggingFace if missing/tiny."""
    if _SOURCE_JSONL.exists():
        records = [
            json.loads(l)
            for l in _SOURCE_JSONL.read_text("utf-8").splitlines() if l.strip()
        ]
        if len(records) >= _SOURCE_MIN:
            return records

    print("[data_pools] Source file missing or too small -- downloading HotpotQA...")
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[data_pools] ERROR: 'datasets' package not installed.\n"
            "  Run: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    _SOURCE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"[data_pools] HotpotQA validation: {len(dataset):,} questions total")

    def _keep(ex: dict) -> bool:
        return (
            ex.get("level", "") in ("hard", "medium")
            and ex.get("type", "") in ("bridge", "comparison")
        )

    filtered = [ex for ex in dataset if _keep(ex)]
    print(f"[data_pools] Filtered to {len(filtered):,} hard/medium bridge+comparison")

    records = []
    for i, ex in enumerate(filtered):
        sup_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))
        records.append({
            "id":                ex.get("id", f"hpqa_{i:05d}"),
            "question":          ex["question"],
            "answer":            ex["answer"],
            "type":              ex["type"],
            "level":             ex["level"],
            "supporting_titles": sup_titles,
        })

    with _SOURCE_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[data_pools] Saved {len(records):,} questions -> {_SOURCE_JSONL}")
    return records


# ── Pool building ──────────────────────────────────────────────────────────────

def build_pools(
    pool_size: int = TRAIN_POOL_SIZE,
    eval_size: int = EVAL_POOL_SIZE,
    seed:      int = POOL_SEED,
    force:     bool = False,
) -> tuple[list[dict], list[dict]]:
    """Sample and persist fixed train + eval pools.

    Parameters
    ----------
    pool_size : Number of questions in the training pool.
    eval_size : Number of questions in the held-out eval pool.
    seed      : RNG seed -- must stay fixed after first build.
    force     : Rebuild even if pool files already exist.

    Returns (train_pool, eval_pool). Zero overlap enforced by question ID.
    """
    if not force and TRAIN_POOL_PATH.exists() and EVAL_POOL_PATH.exists():
        try:
            stored_t = json.loads(TRAIN_POOL_PATH.read_text("utf-8"))
            stored_e = json.loads(EVAL_POOL_PATH.read_text("utf-8"))
            tm = stored_t.get("metadata", {})
            if (
                tm.get("pool_size") == pool_size
                and tm.get("eval_size") == eval_size
                and tm.get("seed") == seed
            ):
                print(
                    f"[data_pools] Pools already built: "
                    f"{len(stored_t['questions'])} train, "
                    f"{len(stored_e['questions'])} eval -- skipping rebuild."
                )
                return stored_t["questions"], stored_e["questions"]
        except (json.JSONDecodeError, KeyError):
            pass
        print("[data_pools] Pool metadata mismatch or corrupt -- rebuilding...")

    source = _load_source()
    if len(source) < pool_size + eval_size:
        ratio     = len(source) / (pool_size + eval_size)
        pool_size = max(50, int(pool_size * ratio))
        eval_size = max(25, int(eval_size * ratio))
        print(
            f"[data_pools] WARNING: only {len(source)} source questions; "
            f"reduced to train={pool_size}, eval={eval_size}.",
            file=sys.stderr,
        )

    rng = random.Random(seed)
    shuffled = source[:]
    rng.shuffle(shuffled)

    train_pool = shuffled[:pool_size]
    eval_pool  = shuffled[pool_size: pool_size + eval_size]

    # Defensive ID-based dedup (should be a no-op after clean shuffle split).
    train_ids = {q["id"] for q in train_pool}
    eval_pool = [q for q in eval_pool if q["id"] not in train_ids]

    TRAIN_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)

    meta = {"pool_size": len(train_pool), "eval_size": len(eval_pool), "seed": seed,
            "source": str(_SOURCE_JSONL),
            "note": "Fixed pool -- do not change seed after first build."}
    TRAIN_POOL_PATH.write_text(
        json.dumps({"metadata": meta, "questions": train_pool}, indent=2, ensure_ascii=False), "utf-8"
    )
    meta_e = {**meta, "note": "Held-out eval pool -- zero overlap with train_pool."}
    EVAL_POOL_PATH.write_text(
        json.dumps({"metadata": meta_e, "questions": eval_pool}, indent=2, ensure_ascii=False), "utf-8"
    )

    bt = sum(1 for q in train_pool if q["type"] == "bridge")
    ct = sum(1 for q in train_pool if q["type"] == "comparison")
    be = sum(1 for q in eval_pool  if q["type"] == "bridge")
    ce = sum(1 for q in eval_pool  if q["type"] == "comparison")
    print(f"[data_pools] Train pool: {len(train_pool)} ({bt} bridge, {ct} comparison) -> {TRAIN_POOL_PATH.name}")
    print(f"[data_pools] Eval pool : {len(eval_pool)}  ({be} bridge, {ce} comparison) -> {EVAL_POOL_PATH.name}")

    overlap = train_ids & {q["id"] for q in eval_pool}
    assert not overlap, f"BUG: {len(overlap)} overlapping IDs!"
    print("[data_pools] Zero-overlap check: PASSED")
    return train_pool, eval_pool


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_train_pool(pool_size: int = TRAIN_POOL_SIZE, seed: int = POOL_SEED) -> list[dict]:
    """Load the fixed training pool, building if not yet cached."""
    if not TRAIN_POOL_PATH.exists():
        print("[data_pools] train_pool.json not found -- building pools now...")
        build_pools(pool_size=pool_size, seed=seed)
    try:
        obj = json.loads(TRAIN_POOL_PATH.read_text("utf-8"))
        qs  = obj["questions"]
        print(f"[data_pools] Loaded train pool: {len(qs)} questions")
        return qs
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[data_pools] Corrupt train_pool.json ({e}) -- rebuilding...", file=sys.stderr)
        train, _ = build_pools(pool_size=pool_size, seed=seed, force=True)
        return train


def load_eval_pool(eval_size: int = EVAL_POOL_SIZE, seed: int = POOL_SEED) -> list[dict]:
    """Load the held-out evaluation pool, building if not yet cached."""
    if not EVAL_POOL_PATH.exists():
        print("[data_pools] eval_pool.json not found -- building pools now...")
        build_pools(eval_size=eval_size, seed=seed)
    try:
        obj = json.loads(EVAL_POOL_PATH.read_text("utf-8"))
        qs  = obj["questions"]
        print(f"[data_pools] Loaded eval pool : {len(qs)} questions")
        return qs
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[data_pools] Corrupt eval_pool.json ({e}) -- rebuilding...", file=sys.stderr)
        _, ev = build_pools(eval_size=eval_size, seed=seed, force=True)
        return ev


# ── Epoch sampler ──────────────────────────────────────────────────────────────

class EpochSampler:
    """Samples from a fixed pool without replacement within each epoch.

    Rationale for epoch-based vs. i.i.d. sampling: random.choice() with
    replacement can leave some questions unseen for many episodes when the pool
    is large.  Epoch-based sampling guarantees every question is visited exactly
    once per epoch before any question repeats, giving more even coverage as
    training steps accumulate.  This is the documented choice for the training
    loop.  If pure i.i.d. (random.choice) is ever preferred instead, replace
    EpochSampler.next() calls with random.choice(pool) and remove this class.
    """

    def __init__(self, pool: list[dict], seed: int = POOL_SEED) -> None:
        self._pool  = pool[:]
        self._rng   = random.Random(seed)
        self._buf:  list[dict] = []
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def next(self) -> dict:
        """Return the next question, reshuffling when the epoch is exhausted."""
        if not self._buf:
            self._buf = self._pool[:]
            self._rng.shuffle(self._buf)
            self._epoch += 1
        return self._buf.pop()

    def state_dict(self) -> dict:
        return {"buf": self._buf, "epoch": self._epoch}

    def load_state_dict(self, state: dict) -> None:
        self._buf   = state.get("buf", [])
        self._epoch = state.get("epoch", 0)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build fixed RL-LAG train/eval pools.")
    parser.add_argument("--pool-size", type=int, default=TRAIN_POOL_SIZE)
    parser.add_argument("--eval-size", type=int, default=EVAL_POOL_SIZE)
    parser.add_argument("--seed",      type=int, default=POOL_SEED)
    parser.add_argument("--force",     action="store_true",
                        help="Rebuild even if pools already exist.")
    args = parser.parse_args()
    train, ev = build_pools(args.pool_size, args.eval_size, args.seed, args.force)
    print(f"\nDone. Train: {len(train)}  Eval: {len(ev)}")
    print(f"  {TRAIN_POOL_PATH}")
    print(f"  {EVAL_POOL_PATH}")