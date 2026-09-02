"""Minimal PPO training loop for RL-LAG Track B.

Track B — B3:
  Trains three small MLP policy networks (π^G, π^R, π^C from policies.py) using
  Proximal Policy Optimization with:
    - Clip ratio ε = 0.2         (matches the paper's stated hyperparameter)
    - GAE λ = 0.95               (Generalized Advantage Estimation)
    - Entropy bonus coeff = 0.01 (encourages exploration)
    - Value loss coeff = 0.5
    - Adam optimisers, lr = 3e-4 for all three policies
    - Checkpoint every 50 rollout steps (resumable across Kaggle/Colab sessions)

The LLM (qwen2.5:3b-instruct via Ollama) is completely frozen throughout.

Usage
-----
    # Smoke test (10 steps, CPU, no Ollama needed if questions are cached)
    python train_ppo.py --steps 10 --checkpoint-every 5

    # Full local run (requires Ollama running)
    python train_ppo.py --steps 200

    # Resume from latest checkpoint
    python train_ppo.py --steps 500 --resume

    # Kaggle T4 recommended for steps > 200
    python train_ppo.py --steps 500 --checkpoint-every 50 --resume

Honesty note
------------
This trains on a frozen 3B local model on a 25-question (or modestly expanded)
HotpotQA subset.  Results are a small-scale directional validation of the RL-LAG
architecture, not a reproduction of the paper's 7B / 50k-rollout configuration.
See reward.py docstring for the full disclaimer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "corpus" / "hotpot_questions.jsonl"

# PPO hyperparameters (clip ε matches paper, others are standard)
CLIP_EPS = 0.2
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4
PPO_EPOCHS = 4          # number of gradient steps per rollout batch
GAMMA = 0.99            # discount factor (episodes are single-step, so ≈1)


# ── Question loader ────────────────────────────────────────────────────────────

TRAIN_PATH = ROOT / "corpus" / "hotpot_train.jsonl"   # 500-question focused subset

# Files with fewer questions than this threshold are treated as demo subsets;
# load_questions() will auto-download the full set in that case.
_FULL_HOTPOT_MIN = 1_000


def download_full_hotpotqa(out_path: Path | None = None) -> Path:
    """Download the full HotpotQA bridge+comparison question set from HuggingFace.

    Filters to hard/medium bridge and comparison questions (multi-hop only).
    Also writes corpus/hotpot_corpus.jsonl with all supporting paragraphs
    so the FAISS index can be rebuilt from the full evidence base.

    Returns the path to the written hotpot_questions.jsonl.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[train_ppo] ERROR: 'datasets' package not installed.\n"
            "  Run: pip install datasets\n"
            "  Then re-run training.",
            file=sys.stderr,
        )
        sys.exit(1)

    dest = out_path or QUESTIONS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    corpus_out = dest.parent / "hotpot_corpus.jsonl"

    print("[train_ppo] Downloading HotpotQA from HuggingFace (~200 MB)...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"[train_ppo] HotpotQA validation set: {len(dataset):,} total questions")

    # Keep only genuine multi-hop: hard/medium bridge + comparison
    def _keep(ex: dict) -> bool:
        return (
            ex.get("level", "") in ("hard", "medium")
            and ex.get("type", "") in ("bridge", "comparison")
        )

    filtered = [ex for ex in dataset if _keep(ex)]
    print(f"[train_ppo] Multi-hop (hard/medium bridge+comparison): {len(filtered):,}")

    import random as _rng
    _rng.seed(42)
    _rng.shuffle(filtered)

    # ── Write all questions to hotpot_questions.jsonl ──────────────────────────
    questions: list[dict] = []
    for i, ex in enumerate(filtered):
        supporting_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))
        questions.append({
            "id":                f"q_{i+1:05d}",
            "question":          ex["question"],
            "answer":            ex["answer"],
            "type":              ex["type"],
            "level":             ex["level"],
            "supporting_titles": supporting_titles,
        })

    with dest.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # ── Write supporting paragraphs to hotpot_corpus.jsonl ────────────────────
    seen_texts: set[str] = set()
    paragraphs: list[dict] = []
    para_id = 0
    for ex in filtered:
        ctx = ex["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            text = " ".join(s.strip() for s in sents).strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            paragraphs.append({
                "id":    f"para_{para_id:05d}",
                "title": title,
                "text":  text,
            })
            para_id += 1

    with corpus_out.open("w", encoding="utf-8") as f:
        for p in paragraphs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    bridge_n = sum(1 for q in questions if q["type"] == "bridge")
    comp_n   = sum(1 for q in questions if q["type"] == "comparison")
    print(
        f"[train_ppo] Saved {len(questions):,} questions "
        f"({bridge_n:,} bridge + {comp_n:,} comparison) -> {dest}"
    )
    print(f"[train_ppo] Saved {len(paragraphs):,} paragraphs -> {corpus_out}")
    return dest


def load_questions(
    n: int | None = None,
    path: Path | None = None,
    auto_download: bool = True,
) -> list[tuple[str, str]]:
    """Load (question, gold_answer) pairs from the HotpotQA corpus.

    Parameters
    ----------
    n             : optional cap on number of questions returned
    path          : path to a .jsonl file; defaults to QUESTIONS_PATH.
    auto_download : if True and QUESTIONS_PATH has fewer than _FULL_HOTPOT_MIN
                    questions (demo subset), automatically download the full set.

    Returns a list of (question_text, gold_answer) tuples.
    """
    src = path if path else QUESTIONS_PATH

    # Auto-download when using the default path and the file is missing/tiny
    if auto_download and src == QUESTIONS_PATH:
        if not src.exists():
            print(
                f"[train_ppo] {src.name} not found — downloading full HotpotQA...",
            )
            download_full_hotpotqa(src)
        else:
            existing = sum(1 for l in src.open(encoding="utf-8") if l.strip())
            if existing < _FULL_HOTPOT_MIN:
                print(
                    f"[train_ppo] {src.name} has only {existing} questions "
                    f"(demo subset detected, threshold={_FULL_HOTPOT_MIN}). "
                    "Downloading full HotpotQA now...",
                )
                download_full_hotpotqa(src)

    if not src.exists():
        print(
            f"[train_ppo] WARNING: {src} not found. "
            "Using a single fallback question for smoke-testing.",
            file=sys.stderr,
        )
        return [("Who directed Inception and what year was it released?", "Christopher Nolan")]

    records = [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [(r["question"], r.get("answer", "")) for r in records if "question" in r]
    if n:
        pairs = pairs[:n]
    return pairs



# ── GAE computation ───────────────────────────────────────────────────────────

def compute_gae(
    rewards: list[float],
    values: list[float],
    next_value: float = 0.0,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> tuple[list[float], list[float]]:
    """Compute Generalized Advantage Estimates and discounted returns.

    For single-step episodes (our case), GAE simplifies to:
      advantage = reward - value_estimate

    For multi-step trajectories within an episode, the full recursion applies.
    Returns (advantages, returns).
    """
    advantages = []
    gae = 0.0
    # Iterate backwards through steps
    vals = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t + 1] - vals[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# ── Per-policy PPO update ─────────────────────────────────────────────────────

class _PolicyBatch(NamedTuple):
    obs: torch.Tensor          # (N, obs_dim)
    actions: torch.Tensor      # (N,)
    old_log_probs: torch.Tensor  # (N,)
    advantages: torch.Tensor   # (N,)
    returns: torch.Tensor      # (N,)


def _ppo_update(policy, optimizer, batch: _PolicyBatch, epochs: int = PPO_EPOCHS) -> dict[str, float]:
    """Run PPO gradient updates for one policy.  Returns loss components dict."""
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    adv = batch.advantages
    # Normalize advantages (stabilizes training; skip when batch has only 1 step)
    if adv.numel() > 1 and adv.std() > 1e-8:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(epochs):
        new_log_probs, new_values, entropy = policy.evaluate(batch.obs, batch.actions)

        # PPO clipped policy loss
        ratio = torch.exp(new_log_probs - batch.old_log_probs.detach())
        clip_ratio = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        policy_loss = -torch.min(ratio * adv, clip_ratio * adv).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(new_values, batch.returns.detach())

        # Total loss
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.mean().item()
        n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }


def _build_batch(traj) -> _PolicyBatch | None:
    """Convert a list of Step objects into a _PolicyBatch tensor bundle."""
    from rollout import Step
    steps = [s for s in traj if isinstance(s, Step)]
    if not steps:
        return None
    obs = torch.stack([s.obs for s in steps])
    actions = torch.stack([s.action for s in steps]).float()
    old_log_probs = torch.stack([s.log_prob for s in steps])
    rewards = [s.reward for s in steps]
    values = [s.value.item() for s in steps]
    advantages, returns = compute_gae(rewards, values)
    return _PolicyBatch(
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs.detach(),
        advantages=torch.tensor(advantages, dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    total_steps: int = 200,
    checkpoint_every: int = 50,
    resume: bool = True,
    max_questions: int | None = None,
    device: str = "cpu",
    question_pool_path: Path | None = None,
) -> None:
    """Main PPO training loop.

    Each 'step' is one complete pipeline episode (one question).
    PPO updates happen after every episode (online PPO).

    Parameters
    ----------
    total_steps        : number of rollout episodes to run
    checkpoint_every   : save checkpoint every N steps
    resume             : whether to load the latest checkpoint before starting
    max_questions      : cap on how many questions to sample from (default: all)
    device             : 'cpu' or 'cuda'
    question_pool_path : path to question file (default: full 17k pool).
                         Pass TRAIN_PATH (500 questions) for focused training
                         where each question repeats often, accelerating convergence.
    """
    from policies import (
        GraphEdgePolicy, RetrievalSelectPolicy, ContextKeepPolicy,
        save_checkpoint, load_latest_checkpoint,
    )
    from rollout import run_rollout

    # ── Init policies + optimizers ─────────────────────────────────────────
    pi_G = GraphEdgePolicy().to(device)
    pi_R = RetrievalSelectPolicy().to(device)
    pi_C = ContextKeepPolicy().to(device)
    opt_G = torch.optim.Adam(pi_G.parameters(), lr=LR)
    opt_R = torch.optim.Adam(pi_R.parameters(), lr=LR)
    opt_C = torch.optim.Adam(pi_C.parameters(), lr=LR)

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_step = 0
    reward_history: list[dict] = []
    if resume:
        start_step, reward_history = load_latest_checkpoint(
            pi_G, pi_R, pi_C, opt_G, opt_R, opt_C
        )

    # ── Load questions ─────────────────────────────────────────────────────
    questions = load_questions(max_questions, path=question_pool_path)
    pool_label = (question_pool_path or QUESTIONS_PATH).name
    if not questions:
        print("[train_ppo] ERROR: No questions available. Exiting.")
        return

    print(
        f"\n{'='*60}\n"
        f"  RL-LAG Track B -- PPO Training\n"
        f"  Pool: {pool_label} ({len(questions)} questions)   Device: {device}\n"
        f"  Steps: {start_step} -> {start_step + total_steps}\n"
        f"  Clip eps={CLIP_EPS}  GAE lam={GAE_LAMBDA}  LR={LR}\n"
        f"{'='*60}\n"
    )

    # ── Training loop ──────────────────────────────────────────────────────
    for global_step in range(start_step, start_step + total_steps):
        question, gold_answer = random.choice(questions)

        # ── Rollout ────────────────────────────────────────────────────────
        result = run_rollout(question, pi_G, pi_R, pi_C, gold_answer=gold_answer)

        if result.error:
            print(f"  [step {global_step:04d}] SKIP (error: {result.error[:60]})")
            continue

        # ── PPO updates ────────────────────────────────────────────────────
        losses: dict[str, dict] = {}
        for name, policy, optimizer, traj in [
            ("G", pi_G, opt_G, result.traj_G),
            ("R", pi_R, opt_R, result.traj_R),
            ("C", pi_C, opt_C, result.traj_C),
        ]:
            batch = _build_batch(traj)
            if batch is not None:
                losses[name] = _ppo_update(policy, optimizer, batch)
            else:
                losses[name] = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # ── Logging ────────────────────────────────────────────────────────
        comps = result.reward_components
        log_entry = {
            "step": global_step,
            "reward": result.reward_score,
            "components": comps,
            "n_steps_G": len(result.traj_G),
            "n_steps_R": len(result.traj_R),
            "n_steps_C": len(result.traj_C),
            "losses": losses,
            "duration_s": result.duration_s,
        }
        reward_history.append(log_entry)

        print(
            f"  [step {global_step:04d}] "
            f"reward={result.reward_score:+.4f}  "
            f"co={comps.get('correctness',0):.2f}  "
            f"rp={comps.get('retrieval_presence',0):.2f}  "
            f"te={comps.get('token_efficiency',0):.2f}  "
            f"lc={comps.get('logical_consistency',0):.2f}  "
            f"gr={comps.get('grounding',0):.2f}  "
            f"G_loss={losses['G']['policy_loss']:.4f}  "
            f"R_loss={losses['R']['policy_loss']:.4f}  "
            f"C_loss={losses['C']['policy_loss']:.4f}  "
            f"({result.duration_s:.1f}s)"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        if (global_step + 1) % checkpoint_every == 0:
            save_checkpoint(
                global_step + 1,
                pi_G, pi_R, pi_C,
                opt_G, opt_R, opt_C,
                reward_history,
            )

    # Final checkpoint
    save_checkpoint(
        start_step + total_steps,
        pi_G, pi_R, pi_C,
        opt_G, opt_R, opt_C,
        reward_history,
    )

    # Summary
    if reward_history:
        scores = [e["reward"] for e in reward_history]
        print(
            f"\n{'='*60}\n"
            f"  Training complete.\n"
            f"  Steps run : {len(scores)}\n"
            f"  Mean reward: {sum(scores)/len(scores):+.4f}\n"
            f"  Best reward: {max(scores):+.4f}\n"
            f"  Checkpoints: checkpoints/\n"
            f"{'='*60}\n"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

# ── Dry-run mock helpers ───────────────────────────────────────────────────────

_DRY_RUN_DECOMP = json.dumps([
    {"id": "q1", "text": "Who directed Inception?", "type": "factual", "depends_on": []},
    {"id": "q2", "text": "What year was Inception released?", "type": "temporal", "depends_on": ["q1"]},
])

def _mock_llm(prompt: str, **kwargs) -> str:  # noqa: ARG001
    """Deterministic stub for call_llm — returns plausible fixed answers."""
    p = prompt.lower()
    if "json" in p or "subproblem" in p:
        return _DRY_RUN_DECOMP
    if "contradict" in p:
        return "NO"
    if "combine" in p or "synthesize" in p:
        return "Christopher Nolan directed Inception, which was released in 2010."
    return "Christopher Nolan (2010)."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL-LAG PPO policy networks (Track B)."
    )
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of rollout episodes (default: 200).")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N steps (default: 50).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh — ignore any existing checkpoints.")
    parser.add_argument("--questions", type=int, default=None,
                        help="Cap on how many questions to sample from (default: all 25).")
    parser.add_argument(
        "--question-pool",
        default="train",
        metavar="POOL",
        help=(
            "Which question pool to train on. Options:\n"
            "  'train' (default) — hotpot_train.jsonl (500 questions, focused);\n"
            "  'full'            — hotpot_questions.jsonl (17,388 questions);\n"
            "  <path>            — any custom .jsonl file.\n"
            "Focused training (train) repeats questions more often so the\n"
            "policy converges faster and accuracy improves sooner."
        ),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Torch device (default: cpu; use cuda on Kaggle T4).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use a mock LLM (no Ollama needed) to smoke-test the PPO loop.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Override checkpoint directory (e.g. /content/drive/MyDrive/rl-lag/checkpoints).")
    args = parser.parse_args()

    # Resolve question pool path
    _pool = args.question_pool.strip().lower()
    if _pool == "train":
        _qpool_path = TRAIN_PATH
    elif _pool == "full":
        _qpool_path = QUESTIONS_PATH
    else:
        _qpool_path = Path(args.question_pool)
    print(f"[train_ppo] Question pool: {_qpool_path.name}")

    # Override checkpoint dir if provided (Colab/Kaggle use case)
    if args.checkpoint_dir:
        import policies as _policies_mod
        _policies_mod.CHECKPOINT_DIR = args.checkpoint_dir
        print(f"[train_ppo] Checkpoint dir overridden -> {args.checkpoint_dir}")

    if args.dry_run:
        print("[train_ppo] DRY-RUN mode: LLM calls replaced with deterministic mock.")
        with patch("llm_client.call_llm", side_effect=_mock_llm), \
             patch("decomposition.call_llm", side_effect=_mock_llm), \
             patch("solver.call_llm", side_effect=_mock_llm):
            train(
                total_steps=args.steps,
                checkpoint_every=args.checkpoint_every,
                resume=not args.no_resume,
                max_questions=args.questions,
                device=args.device,
                question_pool_path=_qpool_path,
            )
    else:
        train(
            total_steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            resume=not args.no_resume,
            max_questions=args.questions,
            device=args.device,
            question_pool_path=_qpool_path,
        )
