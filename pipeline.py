"""End-to-end pipeline orchestration and cache serialization.

Upgrades (Track A):
  A2 — build_graph() now returns (graph, n_components); both values propagated.
  A4 — contradictions_found is collected from solver events and passed to
       compute_reward() instead of the old hardcoded False.
  A6 — completion_tokens from llm_client.get_usage_stats() is forwarded to
       compute_reward() so the token-cost component is real.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

from decomposition import decompose_query
from graph_builder import build_graph
from llm_client import get_usage_stats, reset_usage_stats
from retrieval import retrieve
from reward import compute_reward
from solver import resolve_graph, resolve_graph_stream


def run_pipeline(question: str) -> dict[str, Any]:
    """Blocking, non-streaming run -- used for the cached demo path."""
    reset_usage_stats()          # A6: measure tokens for THIS run only
    subproblems = decompose_query(question)
    graph, n_components = build_graph(subproblems)  # A2
    solved = resolve_graph(graph, retrieve, question=question)
    stats = get_usage_stats()  # A6
    reward = compute_reward(
        solved["retrieved_docs"],
        solved["final_answer"],
        solved.get("contradictions_found", False),  # A4
        completion_tokens=stats.get("completion_tokens", 0),  # A6
    )
    return {
        "question": question,
        "subproblems": subproblems,
        "n_components": n_components,  # A2
        **solved,
        "reward": reward,
    }


def run_pipeline_stream(question: str) -> Generator[dict[str, Any], None, None]:
    """Streaming run for live (non-cached) UI updates.

    Yields, in order:
      1. one {"type": "decomposition", "subproblems": ..., "graph": ..., "n_components": ...} event
      2. one {"type": "node", ...} event per resolved subproblem (includes "contradiction" flag)
      3. one {"type": "final", ...} event with the final answer and reward
    """
    reset_usage_stats()          # A6: measure tokens for THIS run only
    subproblems = decompose_query(question)
    graph, n_components = build_graph(subproblems)  # A2
    yield {
        "type": "decomposition",
        "subproblems": subproblems,
        "graph": graph,
        "n_components": n_components,  # A2
    }

    solved: dict[str, Any] | None = None
    for event in resolve_graph_stream(graph, retrieve, question=question):
        if event["type"] == "node":
            yield event
        else:
            solved = {
                "order": event["order"],
                "retrieved_docs": event["retrieved_docs"],
                "intermediate_answers": event["intermediate_answers"],
                "final_answer": event["final_answer"],
                "contradictions_found": event.get("contradictions_found", False),  # A4
            }

    assert solved is not None
    stats = get_usage_stats()  # A6
    reward = compute_reward(
        solved["retrieved_docs"],
        solved["final_answer"],
        solved["contradictions_found"],  # A4
        completion_tokens=stats.get("completion_tokens", 0),  # A6
    )
    yield {
        "type": "final",
        "question": question,
        "subproblems": subproblems,
        "n_components": n_components,  # A2
        **solved,
        "reward": reward,
    }
