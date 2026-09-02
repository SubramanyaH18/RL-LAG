"""Minimal PPO training loop for RL-LAG Track B.

Track B — B3:
  Trains three small MLP policy networks (pi^G, pi^R, pi^C from policies.py)
  using Proximal Policy Optimization with:
    - Clip ratio eps = 0.2       (matches the paper's stated hyperparameter)
    - GAE lambda = 0.95          (Generalized Advantage Estimation)
    - Entropy bonus coeff = 0.01 (encourages exploration)
    - Value loss coeff = 0.5
    - Adam optimisers, lr = 3e-4 for all three policies
    - Checkpoint every 500 rollout steps (resumable across Colab/Kaggle sessions)

The LLM (qwen2.5:3b-instruct via Ollama) is completely frozen throughout.

Question pool
-------------
Training uses a fixed, seeded pool of 750 hard/medium bridge+comparison questions
from HotpotQA (corpus/train_pool.json), built once by data_pools.py.  A separate
300-question held-out eval pool (corpus/eval_pool.json) with zero overlap is used
for all reported EM/F1 numbers.  Episode sampling is epoch-based (shuffle, exhaust,
reshuffle) for even coverage.  See data_pools.py for config and EpochSampler.

Usage
-----
    # Smoke test (dry-run, no Ollama needed)
    python train_ppo.py --dry-run --steps 5 --skip-sanity

    # Pre-flight sanity check only (recommended before first training run)
    python sanity_check.py

    # Full local run (requires Ollama running, pools built)
    python train_ppo.py --steps 500 --checkpoint-every 100

    # Resume from latest checkpoint (default behaviour)
    python train_ppo.py --steps 1000 --checkpoint-every 500

    # Colab/Kaggle T4 recommended run
    python train_ppo.py --steps 2000 --checkpoint-every 500 --device cuda

Honesty note
------------
Results are a small-scale directional validation of the RL-LAG architecture
on a frozen 3B model, not a reproduction of the paper's 7B / 50k-rollout
configuration.  See reward.py docstring for the full disclaimer.
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


# ── Question pool ─────────────────────────────────────────────────────────────
# Fixed pools are managed by data_pools.py.
# TRAIN_POOL_SIZE = 2000, EVAL_POOL_SIZE = 300 (derived from measured T4 timing).
# Derivation: T4 ceiling ~26,000 eps/session; pool=2000 -> 13 epochs/session.
from data_pools import load_train_pool, EpochSampler, TRAIN_POOL_SIZE, POOL_SEED

# Legacy path kept for --question-pool CLI compat and smoke-tests only.
TRAIN_PATH      = ROOT / "corpus" / "train_pool.json"
QUESTIONS_PATH  = ROOT / "corpus" / "hotpot_questions.jsonl"  # 25-q demo/smoke-test



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
    total_steps:      int  = 26_000,   # T4 session ceiling: ~26,000 eps at 1.6 s/ep
    checkpoint_every: int  = 500,
    resume:           bool = True,
    device:           str  = "cpu",
    pool_size:        int  = TRAIN_POOL_SIZE,
    skip_sanity:      bool = False,
) -> None:
    """Main PPO training loop.

    Each 'step' is one complete pipeline episode (one question).
    PPO updates happen after every episode (online PPO).
    Episode sampling is epoch-based: the fixed pool is shuffled and exhausted
    before reshuffling, giving guaranteed even coverage over each epoch.

    Parameters
    ----------
    total_steps      : number of rollout episodes to run
    checkpoint_every : save checkpoint every N steps (default: 500)
    resume           : load the latest checkpoint before starting
    device           : 'cpu' or 'cuda'
    pool_size        : training pool size; passed to load_train_pool()
    skip_sanity      : if True, skip the pre-flight sanity check
    """
    from policies import (
        GraphEdgePolicy, RetrievalSelectPolicy, ContextKeepPolicy,
        save_checkpoint, load_latest_checkpoint,
    )
    from rollout import run_rollout

    # ── Pre-flight sanity check ────────────────────────────────────────────────
    if not skip_sanity:
        try:
            from sanity_check import run_sanity_check
            run_sanity_check(n_questions=50, verbose=True)
        except Exception as sc_err:
            print(
                f"[train_ppo] WARNING: sanity check raised an error: {sc_err}\n"
                "  Continuing training. Run python sanity_check.py for details.",
                file=sys.stderr,
            )

    # ── Load question pool via EpochSampler ────────────────────────────────────
    pool_records = load_train_pool(pool_size=pool_size)
    if not pool_records:
        print("[train_ppo] ERROR: No questions in train pool. Exiting.")
        return
    # EpochSampler: shuffles pool into epochs (without replacement within each
    # epoch) for even coverage.  If pure i.i.d. sampling is preferred, replace
    # sampler.next() below with random.choice(pool_records).
    sampler = EpochSampler(pool_records, seed=POOL_SEED)

    # ── Init policies + optimizers ─────────────────────────────────────────────
    pi_G = GraphEdgePolicy().to(device)
    pi_R = RetrievalSelectPolicy().to(device)
    pi_C = ContextKeepPolicy().to(device)
    opt_G = torch.optim.Adam(pi_G.parameters(), lr=LR)
    opt_R = torch.optim.Adam(pi_R.parameters(), lr=LR)
    opt_C = torch.optim.Adam(pi_C.parameters(), lr=LR)

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    start_step = 0
    reward_history: list[dict] = []
    if resume:
        start_step, reward_history = load_latest_checkpoint(
            pi_G, pi_R, pi_C, opt_G, opt_R, opt_C
        )
        # Restore sampler state if saved in checkpoint
        import glob
        ckpts = sorted(glob.glob(str(ROOT / "checkpoints" / "checkpoint_step_*.pt")))
        if ckpts:
            try:
                import torch as _torch
                ckpt = _torch.load(ckpts[-1], map_location="cpu")
                if "sampler_state" in ckpt:
                    sampler.load_state_dict(ckpt["sampler_state"])
                    print(f"[train_ppo] Resumed EpochSampler at epoch {sampler.epoch}")
            except Exception:
                pass

    print(
        f"\n{'='*62}\n"
        f"  RL-LAG Track B -- PPO Training\n"
        f"  Pool: train_pool.json ({len(pool_records)} questions)   Device: {device}\n"
        f"  Sampler epoch: {sampler.epoch}   Steps: {start_step} -> {start_step + total_steps}\n"
        f"  Clip eps={CLIP_EPS}  GAE lam={GAE_LAMBDA}  LR={LR}\n"
        f"{'='*62}\n"
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    for global_step in range(start_step, start_step + total_steps):
        q_record = sampler.next()
        question    = q_record["question"]
        gold_answer = q_record.get("answer", "")

        # ── Rollout ────────────────────────────────────────────────────────────
        result = run_rollout(question, pi_G, pi_R, pi_C, gold_answer=gold_answer)

        if result.error:
            print(f"  [step {global_step:04d}] SKIP (error: {result.error[:60]})")
            continue

        # ── PPO updates ────────────────────────────────────────────────────────
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

        # ── Logging ────────────────────────────────────────────────────────────
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
            f"  [step {global_step:04d}|ep{sampler.epoch}] "
            f"reward={result.reward_score:+.4f}  "
            f"co={comps.get('correctness',0):.2f}  "
            f"rp={comps.get('retrieval_presence',0):.2f}  "
            f"te={comps.get('token_efficiency',0):.2f}  "
            f"lc={comps.get('logical_consistency',0):.2f}  "
            f"gr={comps.get('grounding',0):.2f}  "
            f"({result.duration_s:.1f}s)"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if (global_step + 1) % checkpoint_every == 0:
            save_checkpoint(
                global_step + 1,
                pi_G, pi_R, pi_C,
                opt_G, opt_R, opt_C,
                reward_history,
                extra={"sampler_state": sampler.state_dict()},
            )

    # Final checkpoint
    save_checkpoint(
        start_step + total_steps,
        pi_G, pi_R, pi_C,
        opt_G, opt_R, opt_C,
        reward_history,
        extra={"sampler_state": sampler.state_dict()},
    )

    # Summary
    if reward_history:
        scores = [e["reward"] for e in reward_history]
        print(
            f"\n{'='*62}\n"
            f"  Training complete.\n"
            f"  Steps run    : {len(scores)}\n"
            f"  Sampler epoch: {sampler.epoch}\n"
            f"  Mean reward  : {sum(scores)/len(scores):+.4f}\n"
            f"  Best reward  : {max(scores):+.4f}\n"
            f"  Checkpoints  : checkpoints/\n"
            f"{'='*62}\n"
        )


# -- CLI -----------------------------------------------------------------------

# -- Dry-run mock helpers -------------------------------------------------------

_DRY_RUN_DECOMP = json.dumps([
    {"id": "q1", "text": "Who directed Inception?", "type": "factual", "depends_on": []},
    {"id": "q2", "text": "What year was Inception released?", "type": "temporal", "depends_on": ["q1"]},
])

def _mock_llm(prompt: str, **kwargs) -> str:
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
    parser.add_argument("--steps", type=int, default=26_000,
                        help="Number of rollout episodes (default: 26000 = T4 12h ceiling).")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save checkpoint every N steps (default: 500).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh -- ignore any existing checkpoints.")
    parser.add_argument("--pool-size", type=int, default=TRAIN_POOL_SIZE,
                        help=f"Fixed training pool size (default: {TRAIN_POOL_SIZE}).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Torch device (default: cpu; use cuda on Colab/Kaggle T4).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use a mock LLM (no Ollama needed) to smoke-test the PPO loop.")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip the pre-flight sanity check (useful on resume runs).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Override checkpoint directory.")
    args = parser.parse_args()

    if args.checkpoint_dir:
        import policies as _policies_mod
        _policies_mod.CHECKPOINT_DIR = args.checkpoint_dir
        print(f"[train_ppo] Checkpoint dir overridden -> {args.checkpoint_dir}")

    _train_kwargs = dict(
        total_steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        device=args.device,
        pool_size=args.pool_size,
        skip_sanity=args.skip_sanity,
    )

    if args.dry_run:
        print("[train_ppo] DRY-RUN mode: LLM calls replaced with deterministic mock.")
        with patch("llm_client.call_llm", side_effect=_mock_llm), \
             patch("decomposition.call_llm", side_effect=_mock_llm), \
             patch("solver.call_llm", side_effect=_mock_llm):
            train(**_train_kwargs)
    else:
        train(**_train_kwargs)
