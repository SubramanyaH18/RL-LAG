"""Single-episode rollout for RL-LAG Track B PPO training.

Track B — B2:
  Runs the complete pipeline (decompose → graph → retrieve → resolve → reward)
  exactly once per call, intercepting three decision points:

    1. π^G  after build_graph():    keep or drop each proposed edge
    2. π^R  after retrieve():       include or exclude each candidate passage
    3. π^C  after resolve_node():   keep or discard each passage from context

  Records the full trajectory (observations, actions, log-probs, values) needed
  for the PPO update step in train_ppo.py.

  The LLM (qwen2.5:3b-instruct via Ollama) is never updated — only the three policy
  networks receive gradients.

Returns a RolloutResult containing:
  - trajectories for each of the three policies
  - the scalar composite reward + individual components from reward.py (A6)
  - the final answer string (for optional EM/F1 logging)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import networkx as nx

from decomposition import decompose_query
from graph_builder import build_graph
from llm_client import call_llm, get_usage_stats, reset_usage_stats
from policies import (
    GraphEdgePolicy,
    RetrievalSelectPolicy,
    ContextKeepPolicy,
    build_obs_G,
    build_obs_R,
    build_obs_C,
)
from retrieval import get_retriever, retrieve
from reward import compute_reward
from solver import resolve_node, check_contradiction


# ── Trajectory step containers ─────────────────────────────────────────────────

@dataclass
class Step:
    obs: torch.Tensor
    action: torch.Tensor        # 0 or 1
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float = 0.0         # filled in at episode end via credit assignment


@dataclass
class RolloutResult:
    question: str
    final_answer: str
    reward_score: float
    reward_components: dict[str, float]
    # Per-policy trajectories
    traj_G: list[Step] = field(default_factory=list)
    traj_R: list[Step] = field(default_factory=list)
    traj_C: list[Step] = field(default_factory=list)
    # Episode metadata
    n_nodes: int = 0
    n_edges_proposed: int = 0
    n_edges_kept: int = 0
    n_passages_proposed: int = 0
    n_passages_selected: int = 0
    duration_s: float = 0.0
    error: str = ""


# ── Embedding helper (reuses the already-loaded retriever model) ───────────────

def _embed(text: str) -> torch.Tensor:
    """Encode a text string using the shared SentenceTransformer model."""
    retriever = get_retriever()
    emb = retriever.model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return torch.from_numpy(emb[0]).float()   # shape: (384,)


# ── π^G: graph-edge decisions ─────────────────────────────────────────────────

def _apply_graph_policy(
    graph: nx.DiGraph,
    pi_G: GraphEdgePolicy,
    traj_G: list[Step],
) -> nx.DiGraph:
    """For each edge in the graph, ask π^G whether to keep it.

    Dropped edges are removed; the resulting subgraph must remain a DAG
    (it will be, since removing edges from a DAG never creates cycles).
    Isolated nodes (no incoming/outgoing) remain — they become independent roots.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    edges_to_remove: list[tuple[str, str]] = []

    for src, tgt in list(graph.edges()):
        src_type = graph.nodes[src].get("type", "factual")
        tgt_type = graph.nodes[tgt].get("type", "factual")
        obs = build_obs_G(src_type, tgt_type, n_nodes, n_edges)

        with torch.no_grad():
            out = pi_G(obs)

        traj_G.append(Step(
            obs=obs,
            action=out.action.squeeze(),
            log_prob=out.log_prob.squeeze(),
            value=out.value.squeeze(),
        ))

        if out.action.item() == 0:   # 0 = drop
            edges_to_remove.append((src, tgt))

    pruned = graph.copy()
    pruned.remove_edges_from(edges_to_remove)
    return pruned


# ── π^R: passage-selection decisions ─────────────────────────────────────────

def _apply_retrieval_policy(
    subproblem_text: str,
    candidate_docs: list[dict],
    pi_R: RetrievalSelectPolicy,
    traj_R: list[Step],
) -> list[dict]:
    """For each retrieved candidate passage, ask π^R whether to include it."""
    query_emb = _embed(subproblem_text)
    selected: list[dict] = []

    for doc in candidate_docs:
        text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
        passage_emb = _embed(text)
        obs = build_obs_R(passage_emb, query_emb)

        with torch.no_grad():
            out = pi_R(obs)

        traj_R.append(Step(
            obs=obs,
            action=out.action.squeeze(),
            log_prob=out.log_prob.squeeze(),
            value=out.value.squeeze(),
        ))

        if out.action.item() == 1:   # 1 = include
            selected.append(doc)

    # Fallback: if policy drops everything, keep the highest-scored passage
    if not selected and candidate_docs:
        selected = [candidate_docs[0]]

    return selected


# ── π^C: context-keep decisions ───────────────────────────────────────────────

def _apply_context_policy(
    selected_docs: list[dict],
    running_context_tokens: int,
    pi_C: ContextKeepPolicy,
    traj_C: list[Step],
) -> tuple[list[dict], int]:
    """For each selected passage, ask π^C whether to keep it in running context."""
    kept: list[dict] = []
    ctx_tokens = running_context_tokens

    for doc in selected_docs:
        text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
        passage_emb = _embed(text)
        obs = build_obs_C(passage_emb, ctx_tokens)

        with torch.no_grad():
            out = pi_C(obs)

        traj_C.append(Step(
            obs=obs,
            action=out.action.squeeze(),
            log_prob=out.log_prob.squeeze(),
            value=out.value.squeeze(),
        ))

        if out.action.item() == 1:   # 1 = keep
            kept.append(doc)
            ctx_tokens += len(text.split())  # rough token estimate

    # Fallback: keep at least one passage
    if not kept and selected_docs:
        kept = [selected_docs[0]]
        ctx_tokens += len((selected_docs[0].get("text", "") if isinstance(selected_docs[0], dict) else str(selected_docs[0])).split())

    return kept, ctx_tokens


# ── Credit assignment ──────────────────────────────────────────────────────────

def _assign_reward(trajectories: list[Step], scalar_reward: float) -> None:
    """Assign the episode scalar reward to every step (Monte-Carlo return).

    This is the simplest possible credit assignment — every step in the episode
    gets the full episode reward.  For longer training runs, GAE in train_ppo.py
    will refine this using the per-step value estimates.
    """
    for step in trajectories:
        step.reward = scalar_reward


# ── Main rollout function ─────────────────────────────────────────────────────

def run_rollout(
    question: str,
    pi_G: GraphEdgePolicy,
    pi_R: RetrievalSelectPolicy,
    pi_C: ContextKeepPolicy,
    retrieval_k: int = 3,
    gold_answer: str = "",
) -> RolloutResult:
    """Run one complete episode and return the trajectory + reward.

    This is the function called repeatedly by train_ppo.py.  It wraps the
    entire existing pipeline (decompose → graph → retrieve → resolve → reward)
    while inserting policy decisions at the three intervention points.

    The LLM is invoked for:
      - decompose_query()    : 1 call
      - resolve_node()       : 1 call per node
      - check_contradiction(): 1 call per node (from solver.py A4)
      - synthesis            : 1 call
    All LLM calls go through llm_client.py's cache, so repeated questions are free.
    """
    t0 = time.time()
    result = RolloutResult(question=question, final_answer="",
                           reward_score=0.0, reward_components={})
    traj_G: list[Step] = []
    traj_R: list[Step] = []
    traj_C: list[Step] = []

    try:
        reset_usage_stats()   # measure token cost for THIS episode only
        # ── 1. Decompose ──────────────────────────────────────────────────────
        subproblems = decompose_query(question)
        graph, _ = build_graph(subproblems)
        result.n_nodes = graph.number_of_nodes()
        result.n_edges_proposed = graph.number_of_edges()

        # ── 2. π^G — prune edges ──────────────────────────────────────────────
        graph = _apply_graph_policy(graph, pi_G, traj_G)
        result.n_edges_kept = graph.number_of_edges()

        # ── 3. Node resolution with π^R + π^C ────────────────────────────────
        answers: dict[str, str] = {}
        retrieved_docs_per_node: dict[str, list] = {}
        running_ctx_tokens = 0
        any_contradiction = False

        order = list(nx.topological_sort(graph))
        for node_id in order:
            subproblem = dict(graph.nodes[node_id])
            node_type = subproblem.get("type", "factual")

            # Build prior context from dependency answers (A1)
            dep_ids = subproblem.get("depends_on", [])
            prior_context = " ".join(answers[d] for d in dep_ids if d in answers)

            # Retrieve candidates
            candidates = retrieve(
                subproblem["text"],
                k=retrieval_k,
                prior_context=prior_context,
                subproblem_type=node_type,
            )
            result.n_passages_proposed += len(candidates)

            # π^R — select which passages to use
            selected = _apply_retrieval_policy(subproblem["text"], candidates, pi_R, traj_R)
            result.n_passages_selected += len(selected)

            # π^C — keep passages in running context
            kept, running_ctx_tokens = _apply_context_policy(
                selected, running_ctx_tokens, pi_C, traj_C
            )
            retrieved_docs_per_node[node_id] = kept

            # Resolve node with LLM (frozen)
            answer = resolve_node(subproblem, kept, answers)

            # A4 — snapshot BEFORE inserting current answer (same fix as solver.py)
            # so contradiction is checked against prior nodes only, not itself.
            prior_answers_snapshot = dict(answers)
            answers[node_id] = answer
            if check_contradiction(answer, prior_answers_snapshot):
                any_contradiction = True

        # ── 4. Synthesis ──────────────────────────────────────────────────────
        synthesis_prompt = (
            "Combine the intermediate answers below into one direct answer.\n"
            "Use only these answers. Mention uncertainty when present.\n\n"
            + "\n".join(f"{nid}: {answers[nid]}" for nid in order)
        )
        final_answer = call_llm(
            synthesis_prompt,
            system="You synthesize a concise final answer from verified intermediate results.",
            temperature=0.1,
        )
        result.final_answer = final_answer

        # ── 5. Reward (A6 four-component) ─────────────────────────────────────
        stats = get_usage_stats()
        reward_dict = compute_reward(
            retrieved_docs_per_node,
            final_answer,
            any_contradiction,
            completion_tokens=stats.get("completion_tokens", 0),
            gold_answer=gold_answer,
        )
        result.reward_score = reward_dict["score"]
        result.reward_components = reward_dict.get("components", {})

        # ── 6. Assign episode reward to all trajectory steps ──────────────────
        for traj in (traj_G, traj_R, traj_C):
            _assign_reward(traj, result.reward_score)

    except Exception as exc:
        result.error = str(exc)
        print(f"[rollout] error on '{question[:60]}': {exc}")

    result.traj_G = traj_G
    result.traj_R = traj_R
    result.traj_C = traj_C
    result.duration_s = time.time() - t0
    return result


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a single RL-LAG rollout episode.")
    parser.add_argument("--question", default="Who directed Inception and what year was it released?")
    args = parser.parse_args()

    from policies import get_all_policies
    pi_G, pi_R, pi_C = get_all_policies()

    print(f"\n[rollout] question: {args.question}\n")
    r = run_rollout(args.question, pi_G, pi_R, pi_C)

    print(f"\n[rollout] final answer : {r.final_answer}")
    print(f"[rollout] reward score : {r.reward_score:.4f}")
    print(f"[rollout] components   : {r.reward_components}")
    print(f"[rollout] steps G/R/C  : {len(r.traj_G)}/{len(r.traj_R)}/{len(r.traj_C)}")
    print(f"[rollout] duration     : {r.duration_s:.1f}s")
    if r.error:
        print(f"[rollout] error        : {r.error}")
