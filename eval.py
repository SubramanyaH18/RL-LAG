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
QUESTIONS_PATH = ROOT / "corpus" / "hotpot_questions.jsonl"
DEFAULT_OUT = ROOT / "results.json"


# ---------------------------------------------------------------------------
# EM / F1 helpers  (standard HotpotQA evaluation style)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles — matches the official HotpotQA eval."""
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
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Condition (a): zero-shot baseline — top-1 passage retrieval, no LLM
# ---------------------------------------------------------------------------

def run_baseline(question: str) -> str:
    """Retrieve the top passage and return it as the answer (no LLM call)."""
    from retrieval import retrieve
    docs = retrieve(question, k=1)
    if docs:
        return docs[0].get("text", "")
    return ""


# ---------------------------------------------------------------------------
# Condition (c): rule-based pipeline — no PPO policy networks
# ---------------------------------------------------------------------------

def run_pipeline(question: str) -> str:
    """Run the full RL-LAG pipeline and return the final answer string."""
    from pipeline import run_pipeline as _run
    result = _run(question)
    return result.get("final_answer", "")


# ---------------------------------------------------------------------------
# Condition (b): random-init policy networks (freshly instantiated)
# ---------------------------------------------------------------------------

def run_with_random_policies(question: str) -> str:
    """Run the full pipeline with freshly-initialized (random) policy networks.

    This serves as the 'untrained RL baseline' — the policy networks exist but
    have not received any PPO gradient updates.  Comparing against condition (d)
    shows whether training actually improved the policies.
    """
    from policies import get_all_policies
    from rollout import run_rollout

    pi_G, pi_R, pi_C = get_all_policies()
    result = run_rollout(question, pi_G, pi_R, pi_C)
    return result.final_answer


# ---------------------------------------------------------------------------
# Condition (d): PPO-trained policy networks (loaded from checkpoint)
# ---------------------------------------------------------------------------

# Module-level cache so we don't reload the checkpoint for every question.
_trained_policies = None


def _get_trained_policies():
    """Load trained policy networks from the latest checkpoint (cached)."""
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
    # Dummy optimizers required by load_latest_checkpoint API.
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

    pi_G.eval()
    pi_R.eval()
    pi_C.eval()
    _trained_policies = (pi_G, pi_R, pi_C, step)
    return _trained_policies


def run_with_trained_policies(question: str) -> str:
    """Run the full pipeline with PPO-trained policy networks from checkpoint."""
    from rollout import run_rollout

    pi_G, pi_R, pi_C, _ = _get_trained_policies()
    result = run_rollout(question, pi_G, pi_R, pi_C)
    return result.final_answer


# ---------------------------------------------------------------------------
# Question loader
# ---------------------------------------------------------------------------

def load_questions(path: Path, n: int | None) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if n is not None:
        records = records[:n]
    return records


# ---------------------------------------------------------------------------
# Evaluation loop — supports 4-way comparison
# ---------------------------------------------------------------------------

# Registered condition runners — maps condition name to (runner_fn, needs_llm).
_CONDITIONS = {
    "zero_shot_baseline":  (run_baseline, False),
    "random_init_policy":  (run_with_random_policies, True),
    "rule_based_pipeline": (run_pipeline, True),
    "ppo_trained_policy":  (run_with_trained_policies, True),
}


def evaluate(
    questions: list[dict],
    conditions: list[str] | None = None,
) -> dict:
    """Run EM/F1 evaluation across one or more conditions.

    Parameters
    ----------
    questions : list[dict]
        Question records with at least 'question' and 'answer' fields.
    conditions : list[str] or None
        Which conditions to evaluate.  None = all four.
        Valid names: zero_shot_baseline, random_init_policy,
                     rule_based_pipeline, ppo_trained_policy.
    """
    if conditions is None:
        conditions = list(_CONDITIONS.keys())

    per_question: list[dict] = []
    totals: dict[str, dict[str, float]] = {
        c: {"em": 0.0, "f1": 0.0} for c in conditions
    }

    n = len(questions)
    for idx, q in enumerate(questions, start=1):
        question_text = q.get("question", "")
        gold = q.get("answer", "")
        q_type = q.get("type", "unknown")
        q_level = q.get("level", "unknown")

        print(f"[{idx}/{n}] {question_text[:70]}...", flush=True)

        entry: dict = {
            "question": question_text,
            "gold": gold,
            "type": q_type,
            "level": q_level,
        }

        for cond in conditions:
            runner_fn, _ = _CONDITIONS[cond]
            try:
                pred = runner_fn(question_text)
                em = exact_match(pred, gold)
                f1 = f1_score(pred, gold)
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

    # Metadata for honest reporting.
    metadata = {
        "model": "qwen2.5:3b-instruct (frozen, via Ollama)",
        "corpus_passages": 87590,
        "note": (
            "Small-scale directional validation.  Trained on a frozen 3B model, "
            "a few hundred to ~1,000 rollouts on a HotpotQA subset, not the "
            "original paper's 7B backbone / 50k rollouts / 21M-passage index."
        ),
    }

    # Add training step count if we loaded a checkpoint.
    if "ppo_trained_policy" in conditions:
        try:
            _, _, _, step = _get_trained_policies()
            metadata["training_steps"] = step
        except Exception:
            pass

    return {"summary": summary, "per_question": per_question, "metadata": metadata}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_EVAL_MODES = {
    "all": list(_CONDITIONS.keys()),
    "baseline-only": ["zero_shot_baseline"],
    "baseline-pipeline": ["zero_shot_baseline", "rule_based_pipeline"],
    "trained-only": ["ppo_trained_policy"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RL-LAG pipeline against the HotpotQA subset."
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N questions (default: all).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSON file (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline retrieval; skip the full pipeline. "
             "(Legacy flag — prefer --eval-mode baseline-only.)",
    )
    parser.add_argument(
        "--eval-mode",
        choices=list(_EVAL_MODES.keys()),
        default=None,
        help="Which conditions to evaluate (default: 'all').",
    )
    args = parser.parse_args()

    if not QUESTIONS_PATH.exists():
        print(
            f"ERROR: {QUESTIONS_PATH} not found.\n"
            "Run: python scripts/build_hotpot_subset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve conditions.
    if args.baseline_only:
        conditions = ["zero_shot_baseline"]
    elif args.eval_mode:
        conditions = _EVAL_MODES[args.eval_mode]
    else:
        conditions = _EVAL_MODES["all"]

    questions = load_questions(QUESTIONS_PATH, args.questions)
    print(
        f"Evaluating {len(questions)} questions — "
        f"conditions: {', '.join(conditions)}\n"
    )

    t0 = time.time()
    results = evaluate(questions, conditions=conditions)
    elapsed = time.time() - t0

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {args.out}  ({elapsed:.1f}s)")

    # Print summary table.
    print("\n--- Summary ------------------------------------")
    for system, metrics in results["summary"].items():
        print(
            f"  {system:<24}  EM={metrics['em']:.4f}  F1={metrics['f1']:.4f}"
            f"  (n={metrics['n']})"
        )
    print("------------------------------------------")


if __name__ == "__main__":
    main()
