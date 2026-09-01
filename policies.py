"""Three lightweight actor-critic MLP policy networks for RL-LAG Track B PPO.

Architecture (Track B — B1):
  π^G  graph-edge policy     — decides keep/drop for each LLM-proposed DAG edge
  π^R  retrieval-select policy — decides include/exclude for each retrieved passage
  π^C  context-keep policy   — decides keep/discard for passage in running context

Design constraints (from roadmap):
  - Hidden size 256–512, matching the paper's own architecture choice
  - 2–3 layer MLP with ReLU activations
  - Bernoulli action space for each binary decision
  - Actor + critic heads share the same trunk (standard A2C/PPO pattern)
  - All embeddings use all-MiniLM-L6-v2 (384-dim) — already loaded by retrieval.py

Observation spaces:
  π^G  : [src_type_onehot(4), tgt_type_onehot(4), n_nodes_norm(1), n_edges_norm(1)]
          total dim = 10
  π^R  : [passage_emb(384), query_emb(384)]   total dim = 768
  π^C  : [passage_emb(384), ctx_len_norm(1)]  total dim = 385

Honesty note (carried over from reward.py):
  These networks are trained on a frozen 3B local model (qwen2.5:3b-instruct via Ollama),
  not the paper's fine-tuned 7B backbone.  Training targets a few hundred to
  ~1,000 rollouts on a 25-question HotpotQA subset, not 50,000 rollouts on 21M
  Wikipedia passages.  Results are a small-scale directional validation of the
  RL-LAG optimization layer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

# ── Constants ──────────────────────────────────────────────────────────────────
EMB_DIM = 384          # all-MiniLM-L6-v2 output dimension
MAX_NODES = 6          # == MAX_SUBPROBLEMS (never increased per roadmap)
MAX_EDGES = 15         # C(6,2) upper bound for a 6-node DAG
SUBPROBLEM_TYPES = ["factual", "relational", "comparative", "temporal"]
N_TYPES = len(SUBPROBLEM_TYPES)

# Observation dimensions
OBS_DIM_G = N_TYPES + N_TYPES + 1 + 1   # 10
OBS_DIM_R = EMB_DIM + EMB_DIM            # 768
OBS_DIM_C = EMB_DIM + 1                  # 385


# ── Shared MLP trunk builder ───────────────────────────────────────────────────

def _make_trunk(in_dim: int, hidden: int, n_layers: int = 2) -> nn.Sequential:
    """Build a shared feature-extraction trunk (n_layers ReLU layers)."""
    layers: list[nn.Module] = []
    for i in range(n_layers):
        layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.ReLU()]
    return nn.Sequential(*layers)


# ── Policy output container ────────────────────────────────────────────────────

class PolicyOutput(NamedTuple):
    action: torch.Tensor        # int tensor, shape (batch,) — 0 or 1
    log_prob: torch.Tensor      # shape (batch,)
    entropy: torch.Tensor       # shape (batch,)
    value: torch.Tensor         # shape (batch,) — critic estimate
    dist: Bernoulli             # the distribution (for PPO ratio re-computation)


# ── Base actor-critic ──────────────────────────────────────────────────────────

class ActorCriticMLP(nn.Module):
    """Generic binary actor-critic MLP.  Subclasses set obs_dim and hidden."""

    obs_dim: int
    hidden: int

    def __init__(self) -> None:
        super().__init__()
        self.trunk = _make_trunk(self.obs_dim, self.hidden, n_layers=2)
        self.actor_head = nn.Linear(self.hidden, 1)   # logit for Bernoulli
        self.critic_head = nn.Linear(self.hidden, 1)  # state value

    def forward(self, obs: torch.Tensor) -> PolicyOutput:
        """Forward pass.  obs shape: (batch, obs_dim) or (obs_dim,)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.trunk(obs)                          # (batch, hidden)
        logit = self.actor_head(features).squeeze(-1)       # (batch,)
        value = self.critic_head(features).squeeze(-1)      # (batch,)
        dist = Bernoulli(logits=logit)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            dist=dist,
        )

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-compute log_prob, value, entropy for stored (obs, action) pairs.

        Used in the PPO update step when iterating over the replay buffer.
        Returns (log_prob, value, entropy).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            pass  # already (batch,)
        features = self.trunk(obs)
        logit = self.actor_head(features).squeeze(-1)
        value = self.critic_head(features).squeeze(-1)
        dist = Bernoulli(logits=logit)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy


# ── The three policy networks ──────────────────────────────────────────────────

class GraphEdgePolicy(ActorCriticMLP):
    """π^G — decides whether to keep or drop each LLM-proposed DAG edge.

    Observation (10 dims):
      src_type_onehot : 4 dims  (factual / relational / comparative / temporal)
      tgt_type_onehot : 4 dims
      n_nodes_norm    : 1 dim   (n_nodes / MAX_NODES)
      n_edges_norm    : 1 dim   (n_edges / MAX_EDGES)

    Action: 1 = keep edge, 0 = drop edge (node becomes a root).
    """
    obs_dim = OBS_DIM_G
    hidden = 256


class RetrievalSelectPolicy(ActorCriticMLP):
    """π^R — decides which retrieved passages to include for a node.

    Observation (768 dims):
      passage_emb : 384 dims  (SentenceTransformer embedding of passage text)
      query_emb   : 384 dims  (embedding of the subproblem text)

    Action: 1 = include passage, 0 = exclude.
    """
    obs_dim = OBS_DIM_R
    hidden = 512


class ContextKeepPolicy(ActorCriticMLP):
    """π^C — decides whether to keep a passage in the running context.

    Observation (385 dims):
      passage_emb  : 384 dims
      ctx_len_norm :   1 dim  (running context token count / 2048 budget)

    Action: 1 = keep in context, 0 = discard.
    """
    obs_dim = OBS_DIM_C
    hidden = 256


# ── Observation builders ───────────────────────────────────────────────────────

def type_onehot(subproblem_type: str) -> torch.Tensor:
    """Convert a subproblem type string to a 4-dim one-hot tensor."""
    idx = SUBPROBLEM_TYPES.index(subproblem_type) if subproblem_type in SUBPROBLEM_TYPES else 0
    vec = torch.zeros(N_TYPES)
    vec[idx] = 1.0
    return vec


def build_obs_G(
    src_type: str,
    tgt_type: str,
    n_nodes: int,
    n_edges: int,
) -> torch.Tensor:
    """Build a π^G observation vector."""
    return torch.cat([
        type_onehot(src_type),
        type_onehot(tgt_type),
        torch.tensor([n_nodes / MAX_NODES, n_edges / max(MAX_EDGES, 1)]),
    ])  # shape: (10,)


def build_obs_R(passage_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
    """Build a π^R observation vector."""
    return torch.cat([passage_emb, query_emb])  # shape: (768,)


def build_obs_C(passage_emb: torch.Tensor, ctx_tokens: int, budget: int = 2048) -> torch.Tensor:
    """Build a π^C observation vector."""
    ctx_norm = torch.tensor([min(ctx_tokens / budget, 1.0)])
    return torch.cat([passage_emb, ctx_norm])  # shape: (385,)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def save_checkpoint(
    step: int,
    pi_G: GraphEdgePolicy,
    pi_R: RetrievalSelectPolicy,
    pi_C: ContextKeepPolicy,
    opt_G: torch.optim.Optimizer,
    opt_R: torch.optim.Optimizer,
    opt_C: torch.optim.Optimizer,
    reward_history: list[dict],
) -> Path:
    """Save all three policy + optimizer states to checkpoints/checkpoint_step_NNN.pt."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"checkpoint_step_{step:04d}.pt"
    torch.save(
        {
            "step": step,
            "pi_G_state": pi_G.state_dict(),
            "pi_R_state": pi_R.state_dict(),
            "pi_C_state": pi_C.state_dict(),
            "opt_G_state": opt_G.state_dict(),
            "opt_R_state": opt_R.state_dict(),
            "opt_C_state": opt_C.state_dict(),
            "reward_history": reward_history,
        },
        path,
    )
    print(f"[checkpoint] saved -> {path}")
    return path


def load_latest_checkpoint(
    pi_G: GraphEdgePolicy,
    pi_R: RetrievalSelectPolicy,
    pi_C: ContextKeepPolicy,
    opt_G: torch.optim.Optimizer,
    opt_R: torch.optim.Optimizer,
    opt_C: torch.optim.Optimizer,
) -> tuple[int, list[dict]]:
    """Load the most recent checkpoint if one exists.  Returns (step, reward_history)."""
    if not CHECKPOINT_DIR.exists():
        return 0, []
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return 0, []
    path = checkpoints[-1]
    data = torch.load(path, map_location="cpu", weights_only=False)
    pi_G.load_state_dict(data["pi_G_state"])
    pi_R.load_state_dict(data["pi_R_state"])
    pi_C.load_state_dict(data["pi_C_state"])
    opt_G.load_state_dict(data["opt_G_state"])
    opt_R.load_state_dict(data["opt_R_state"])
    opt_C.load_state_dict(data["opt_C_state"])
    step = data["step"]
    history = data.get("reward_history", [])
    print(f"[checkpoint] resumed from step {step} <- {path}")
    return step, history


def get_all_policies() -> tuple[GraphEdgePolicy, RetrievalSelectPolicy, ContextKeepPolicy]:
    """Convenience factory — creates all three policy networks."""
    return GraphEdgePolicy(), RetrievalSelectPolicy(), ContextKeepPolicy()
