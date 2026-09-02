"""EM/F1 evaluation harness for the RL-LAG prototype — 4-way comparison.

Track A — A7 (original): Compare the full RL-LAG pipeline against a naive
baseline.

Track B — Step 4 (extended): Four-way comparison across all conditions:
  (a) zero_shot_baseline   — top-1 passage retrieval, no LLM, no decomposition
  (b) random_init_policy   — full pipeline with freshly-initialized (random)
                             policy networks (π^G, π^R, π^C)
  (c) rule_based_pipeline  — full pipeline without PPO policy networks
  (d) ppo_trained_policy   — full pipeline with PPO-trained policy networks
                             loaded from the latest checkpoint

Usage
-----
    # Run all four conditions (requires Ollama + checkpoint)
    python eval.py --questions 100 --eval-mode all

    # Run only baseline + pipeline (original A7 behavior)
    python eval.py --questions 100 --eval-mode baseline-pipeline

    # Run baseline only (no LLM needed)
    python eval.py --questions 100 --baseline-only

Output
------
results.json contains:
  {
    "summary": {
        "zero_shot_baseline":   {"em": float, "f1": float, "n": int},
        "random_init_policy":   {"em": float, "f1": float, "n": int},
        "rule_based_pipeline":  {"em": float, "f1": float, "n": int},
        "ppo_trained_policy":   {"em": float, "f1": float, "n": int}
    },
    "per_question": [ {...}, ... ],
    "metadata": { ... }
  }

Honesty note
------------
EM/F1 numbers are measured on a HotpotQA subset using a locally-run frozen 3B
model (qwen2.5:3b-instruct via Ollama), not the paper's fine-tuned 7B model
evaluated over the full dev set.  Training used a few hundred to ~1,000
rollouts on this same subset, not the original paper's 50k rollouts over 21M
Wikipedia passages.  Results should be interpreted as a small-scale directional
validation of the RL-LAG architecture, not a head-to-head with the published
numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Reported results always use the held-out eval pool (300 questions, zero
# overlap with the training pool).  The 25-question demo file is kept only
# as a fast smoke-test and is NOT used for any reported numbers.
EVAL_POOL_PATH  = ROOT / "corpus" / "eval_pool.json"
SMOKE_TEST_PATH = ROOT / "corpus" / "hotpot_questions.jsonl"   # 25-q demo subset
DEFAULT_OUT     = ROOT / "results.json"


# ---------------------------------------------------------------------------
# EM / F1 helpers  (standard HotpotQA evaluation style)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles -- matches the official HotpotQA eval."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Condition (a): zero-shot baseline
# ---------------------------------------------------------------------------

def run_baseline(question: str) -> str:
    from retrieval import retrieve
    docs = retrieve(question, k=1)
    if docs:
        return docs[0].get("text", "")
    return ""


# ---------------------------------------------------------------------------
# Condition (c): rule-based pipeline
# ---------------------------------------------------------------------------

def run_pipeline(question: str) -> str:
    from pipeline import run_pipeline as _run
    result = _run(question)
    return result.get("final_answer", "")


# ---------------------------------------------------------------------------
# Condition (b): random-init policy networks
# ---------------------------------------------------------------------------

def run_with_random_policies(question: str) -> str:
    from policies import get_all_policies
    from rollout import run_rollout
    pi_G, pi_R, pi_C = get_all_policies()
    result = run_rollout(question, pi_G, pi_R, pi_C)
    return result.final_answer


# ---------------------------------------------------------------------------
# Condition (d): PPO-trained policy networks
# ---------------------------------------------------------------------------

_trained_policies = None


def _get_trained_policies():
    global _trained_policies
    if _trained_policies is not None:
        return _trained_policies

    import torch
    from policies import (
        GraphEdgePolicy, RetrievalSelectPolicy, ContextKeepPolicy,
        load_latest_checkpoint,
    )

    pi_G = GraphEdgePolicy()
    pi_R = RetrievalSelectPolicy()
    pi_C = ContextKeepPolicy()
    opt_G = torch.optim.Adam(pi_G.parameters(), lr=1e-4)
    opt_R = torch.optim.Adam(pi_R.parameters(), lr=1e-4)
    opt_C = torch.optim.Adam(pi_C.parameters(), lr=1e-4)

    step, _ = load_latest_checkpoint(pi_G, pi_R, pi_C, opt_G, opt_R, opt_C)
    if step == 0:
        print(
            "[eval] WARNING: No trained checkpoint found. "
            "'ppo_trained_policy' will use random-init weights.",
            file=sys.stderr,
        )

    pi_G.eval(); pi_R.eval(); pi_C.eval()
    _trained_policies = (pi_G, pi_R, pi_C, step)
    return _trained_policies


def run_with_trained_policies(question: str) -> str:
    from rollout import run_rollout
    pi_G, pi_R, pi_C, _ = _get_trained_policies()
    result = run_rollout(question, pi_G, pi_R, pi_C)
    return result.final_answer


# ---------------------------------------------------------------------------
# Question loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, n: int | None) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if n is not None:
        records = records[:n]
    return records


def load_questions(path: Path, n: int | None) -> list[dict]:
    """Back-compat wrapper used by tests."""
    return _load_jsonl(path, n)


def _load_eval_pool(n: int | None) -> list[dict]:
    """Load the held-out eval pool (300 questions). Builds it if missing."""
    if not EVAL_POOL_PATH.exists():
        print("[eval] eval_pool.json not found -- building pools now...")
        from data_pools import build_pools
        build_pools()
    obj = json.loads(EVAL_POOL_PATH.read_text("utf-8"))
    qs  = obj.get("questions", obj) if isinstance(obj, dict) else obj
    if n is not None:
        qs = qs[:n]
    return qs


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _mcnemar(em_a: list[float], em_b: list[float]) -> float:
    """McNemar's test p-value (two-sided) for two paired EM score lists.

    Returns the p-value.  Uses chi-squared approximation (continuity-corrected)
    when b+c >= 25, and the exact binomial otherwise.
    """
    import math
    b = sum(1 for a, bb in zip(em_a, em_b) if a == 1.0 and bb == 0.0)
    c = sum(1 for a, bb in zip(em_a, em_b) if a == 0.0 and bb == 1.0)
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    if n_disc >= 25:
        chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
        # One-tailed survival of chi2 with df=1, approximated via erfc
        p = math.erfc(math.sqrt(chi2 / 2))
        return round(float(p), 6)
    # Exact binomial (two-sided): sum of binom(n, min(b,c)) terms
    smaller = min(b, c)
    p = 0.0
    binom = 1.0
    # Pre-compute binom(n_disc, 0)
    for k in range(n_disc + 1):
        if k == 0:
            binom = 0.5 ** n_disc
        else:
            binom = binom * (n_disc - k + 1) / k
        if k <= smaller or k >= n_disc - smaller:
            p += binom
    return round(min(1.0, p * 2), 6)


def _bootstrap_f1_ci(
    f1_a: list[float],
    f1_b: list[float],
    n_resample: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap resampling for mean F1 difference (B - A).

    Returns {'mean_diff': float, 'ci_low': float, 'ci_high': float, 'p_value': float}.
    ci_low/ci_high are the 95% CI on the mean F1 difference.
    p_value is the fraction of resamples where the difference was <= 0.
    """
    import random as _rng
    _rng.seed(seed)
    n = len(f1_a)
    observed_diff = sum(f1_b) / n - sum(f1_a) / n
    diffs = []
    for _ in range(n_resample):
        indices = [_rng.randint(0, n - 1) for _ in range(n)]
        sample_a = sum(f1_a[i] for i in indices) / n
        sample_b = sum(f1_b[i] for i in indices) / n
        diffs.append(sample_b - sample_a)
    diffs.sort()
    ci_low  = diffs[int(0.025 * n_resample)]
    ci_high = diffs[int(0.975 * n_resample)]
    p_val   = sum(1 for d in diffs if d <= 0.0) / n_resample
    return {
        "mean_diff": round(observed_diff, 4),
        "ci_low":    round(ci_low, 4),
        "ci_high":   round(ci_high, 4),
        "p_value":   round(p_val, 4),
    }


def compute_statistics(per_question: list[dict], conditions: list[str]) -> dict:
    """Compute McNemar's test and bootstrap CI for all condition pairs vs PPO-trained."""
    stats: dict = {}
    reference = "ppo_trained_policy"
    if reference not in conditions:
        return stats

    ref_em = [q.get(reference, {}).get("em", 0.0) for q in per_question]
    ref_f1 = [q.get(reference, {}).get("f1", 0.0) for q in per_question]

    for cond in conditions:
        if cond == reference:
            continue
        cond_em = [q.get(cond, {}).get("em", 0.0) for q in per_question]
        cond_f1 = [q.get(cond, {}).get("f1", 0.0) for q in per_question]
        key = f"{reference}_vs_{cond}"
        stats[key] = {
            "mcnemar_p_value": _mcnemar(ref_em, cond_em),
            "bootstrap_f1":    _bootstrap_f1_ci(cond_f1, ref_f1),
            "note": f"ppo_trained_policy F1 - {cond} F1 (positive = PPO better)",
        }
    return stats


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

_CONDITIONS = {
    "zero_shot_baseline":  (run_baseline,              False),
    "random_init_policy":  (run_with_random_policies,  True),
    "rule_based_pipeline": (run_pipeline,              True),
    "ppo_trained_policy":  (run_with_trained_policies, True),
}


def evaluate(
    questions:  list[dict],
    conditions: list[str] | None = None,
    run_stats:  bool = True,
) -> dict:
    """Run EM/F1 evaluation across one or more conditions.

    Parameters
    ----------
    questions  : list[dict] with at least 'question' and 'answer' fields.
    conditions : Which conditions to evaluate.  None = all four.
    run_stats  : Whether to compute McNemar + bootstrap statistics.
    """
    if conditions is None:
        conditions = list(_CONDITIONS.keys())

    per_question: list[dict] = []
    totals: dict[str, dict[str, float]] = {c: {"em": 0.0, "f1": 0.0} for c in conditions}

    n = len(questions)
    for idx, q in enumerate(questions, start=1):
        question_text = q.get("question", "")
        gold          = q.get("answer", "")
        q_type        = q.get("type", "unknown")
        q_level       = q.get("level", "unknown")

        print(f"[{idx}/{n}] {question_text[:70]}...", flush=True)

        entry: dict = {
            "question": question_text,
            "gold":     gold,
            "type":     q_type,
            "level":    q_level,
        }

        for cond in conditions:
            runner_fn, _ = _CONDITIONS[cond]
            try:
                pred = runner_fn(question_text)
                em   = exact_match(pred, gold)
                f1   = f1_score(pred, gold)
            except Exception as exc:
                print(f"  {cond} error: {exc}", flush=True)
                pred = f"ERROR: {exc}"
                em, f1 = 0.0, 0.0

            totals[cond]["em"] += em
            totals[cond]["f1"] += f1
            entry[cond] = {"prediction": pred, "em": em, "f1": f1}

            print(f"  {cond:<24} EM={em:.0f} F1={f1:.2f}", flush=True)

        per_question.append(entry)

    summary = {
        cond: {
            "em": round(totals[cond]["em"] / n, 4) if n else 0.0,
            "f1": round(totals[cond]["f1"] / n, 4) if n else 0.0,
            "n": n,
        }
        for cond in conditions
    }

    metadata = {
        "model": "qwen2.5:3b-instruct (frozen, via Ollama)",
        "eval_pool": "eval_pool.json (300 held-out questions, zero overlap with train_pool)",
        "note": (
            "Small-scale directional validation. Trained on a frozen 3B model "
            "with epoch-based sampling over a fixed 750-question train pool. "
            "Not comparable to the paper's 7B / 50k-rollout / 21M-passage configuration."
        ),
    }

    if "ppo_trained_policy" in conditions:
        try:
            _, _, _, step = _get_trained_policies()
            metadata["training_steps"] = step
        except Exception:
            pass

    statistics: dict = {}
    if run_stats and len(conditions) > 1:
        print("\n[eval] Computing McNemar + bootstrap statistics...", flush=True)
        statistics = compute_statistics(per_question, conditions)

    return {
        "summary":     summary,
        "per_question": per_question,
        "statistics":  statistics,
        "metadata":    metadata,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EVAL_MODES = {
    "all":              list(_CONDITIONS.keys()),
    "baseline-only":    ["zero_shot_baseline"],
    "baseline-pipeline": ["zero_shot_baseline", "rule_based_pipeline"],
    "trained-only":     ["ppo_trained_policy"],
    "ppo-only":         ["ppo_trained_policy"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RL-LAG pipeline against the HotpotQA eval pool."
    )
    parser.add_argument(
        "--questions", type=int, default=None, metavar="N",
        help="Evaluate only the first N questions (default: all in pool).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output JSON file (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run only the baseline retrieval. (Legacy flag -- prefer --eval-mode baseline-only.)",
    )
    parser.add_argument(
        "--eval-mode", choices=list(_EVAL_MODES.keys()), default=None,
        help="Which conditions to evaluate (default: all).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=(
            "Use the 25-question demo subset (hotpot_questions.jsonl) instead of "
            "eval_pool.json.  Fast, but NOT used for any reported results."
        ),
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Skip McNemar / bootstrap statistics (faster).",
    )
    args = parser.parse_args()

    if args.smoke_test:
        if not SMOKE_TEST_PATH.exists():
            print(
                f"ERROR: {SMOKE_TEST_PATH} not found.\n"
                "Run: python scripts/build_hotpot_subset.py",
                file=sys.stderr,
            )
            sys.exit(1)
        questions = _load_jsonl(SMOKE_TEST_PATH, args.questions)
        print(
            f"[eval] SMOKE-TEST mode: {len(questions)} questions from "
            f"{SMOKE_TEST_PATH.name}  (not for reported results)"
        )
    else:
        questions = _load_eval_pool(args.questions)
        print(f"[eval] Eval pool: {len(questions)} questions from {EVAL_POOL_PATH.name}")

    if args.baseline_only:
        conditions = ["zero_shot_baseline"]
    elif args.eval_mode:
        conditions = _EVAL_MODES[args.eval_mode]
    else:
        conditions = _EVAL_MODES["all"]

    print(f"Conditions: {', '.join(conditions)}\n")

    t0 = time.time()
    results = evaluate(questions, conditions=conditions, run_stats=not args.no_stats)
    elapsed = time.time() - t0

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {args.out}  ({elapsed:.1f}s)")

    print("\n--- Summary -----------------------------------------------")
    for system, metrics in results["summary"].items():
        print(
            f"  {system:<24}  EM={metrics['em']:.4f}  F1={metrics['f1']:.4f}"
            f"  (n={metrics['n']})"
        )

    if results.get("statistics"):
        print("\n--- Statistical Tests (PPO-trained vs others) ------------")
        for pair, s in results["statistics"].items():
            bs = s["bootstrap_f1"]
            print(
                f"  {pair}\n"
                f"    McNemar p = {s['mcnemar_p_value']:.4f}\n"
                f"    F1 diff   = {bs['mean_diff']:+.4f}  "
                f"95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]  "
                f"bootstrap p = {bs['p_value']:.4f}"
            )
    print("-----------------------------------------------------------")


if __name__ == "__main__":
    main()



# ---------------------------------------------------------------------------
# EM / F1 helpers  (standard HotpotQA evaluation style)
# ---------------------------------------------------------------------------

