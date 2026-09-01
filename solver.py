"""Resolve DAG nodes in topological order and synthesize the final answer.

Upgrades (Track A):
  A1 — retrieval_fn is now called with prior_context (accumulated prior answers)
       so each node's retrieval is conditioned on its dependency results.
  A3 — subproblem type is forwarded to retrieval_fn for type-aware k/scoring.
  A4 — check_contradiction() uses one extra call_llm() per node to detect logical
       conflicts between a new answer and previously resolved answers; the flag is
       yielded in every node event and aggregated for pipeline.py.
"""
from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import networkx as nx

from llm_client import call_llm


def _doc_text(doc: Any) -> str:
    """Docs may be plain strings (old cache format) or {"text","score"} dicts
    (current retrieval.py output) -- normalize to text for prompting."""
    if isinstance(doc, dict):
        return doc.get("text", "")
    return str(doc)


# ---------------------------------------------------------------------------
# A4 — Contradiction checker
# ---------------------------------------------------------------------------

def check_contradiction(
    node_answer: str,
    prior_answers: dict[str, str],
) -> bool:
    """Return True if node_answer logically contradicts any prior answer.

    Uses one extra call_llm() call. Returns False when prior_answers is empty
    (nothing to contradict) or when the LLM judge finds no conflict.

    NOTE: This is one extra LLM call per node. For the 3B local model the
    latency is small; the result is cached by llm_client so identical
    (node_answer, prior_answers) pairs are free on subsequent calls.
    """
    if not prior_answers:
        return False

    prior_block = "\n".join(
        f"- {key}: {answer}" for key, answer in prior_answers.items()
    )
    prompt = (
        "You are a logical consistency checker.\n\n"
        "New statement:\n"
        f"{node_answer}\n\n"
        "Previously established facts:\n"
        f"{prior_block}\n\n"
        "Does the new statement directly contradict any of the previously "
        "established facts? Answer with a single word: YES or NO."
    )
    response = call_llm(
        prompt,
        system="Respond with only YES or NO.",
        temperature=0.0,
        max_tokens=4,
    )
    return response.strip().upper().startswith("YES")


# ---------------------------------------------------------------------------
# Node resolver
# ---------------------------------------------------------------------------

def resolve_node(
    subproblem: dict,
    retrieved_docs: list,
    prior_answers: dict[str, str],
) -> str:
    dependencies = subproblem.get("depends_on", [])
    relevant_prior = {key: prior_answers[key] for key in dependencies if key in prior_answers}
    evidence_lines = [_doc_text(doc) for doc in retrieved_docs]
    prompt = f"""Answer the atomic subproblem using only the supplied evidence and dependency answers.
If the evidence is insufficient, say so in one sentence. Do not invent facts.

Subproblem: {subproblem['text']}
Type: {subproblem.get('type', 'factual')}
Dependency answers: {relevant_prior or 'None'}
Retrieved evidence:
{chr(10).join(f'- {text}' for text in evidence_lines) or '- No evidence retrieved'}

Return ONLY the direct answer in as few words as possible. No explanation."""
    return call_llm(
        prompt,
        system=(
            "You are a precise, evidence-grounded assistant. "
            "Give direct factual answers. Never add sentences or explanations around the answer."
        ),
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Streaming resolver (A1 + A3 + A4 wired in)
# ---------------------------------------------------------------------------

def resolve_graph_stream(
    graph: nx.DiGraph,
    retrieval_fn: Callable[..., list],
    solver_fn: Callable[[dict, list, dict[str, str]], str] = resolve_node,
    question: str = "",
) -> Generator[dict[str, Any], None, None]:
    """Resolve nodes in topological order, yielding one event per node as it
    completes, then a final event with the synthesized answer. Lets the UI
    show live "resolving -> resolved" status instead of a static list.

    Each node event now includes:
      - 'contradiction': bool — whether this node's answer contradicts prior answers (A4)

    The final event includes:
      - 'contradictions_found': bool — True if any node flagged a contradiction (A4)
    """
    answers: dict[str, str] = {}
    retrieved: dict[str, list] = {}
    order = list(nx.topological_sort(graph))
    any_contradiction = False

    for node_id in order:
        subproblem = dict(graph.nodes[node_id])
        node_type = subproblem.get("type", "factual")

        # A1 — build prior context from dependency answers.
        dep_ids = subproblem.get("depends_on", [])
        prior_context = " ".join(
            answers[dep] for dep in dep_ids if dep in answers
        )

        # A1 + A3 — retrieval conditioned on prior context and subproblem type.
        try:
            docs = retrieval_fn(
                subproblem["text"],
                2,
                prior_context=prior_context,
                subproblem_type=node_type,
            )
        except TypeError:
            # Fallback for retrieval_fn signatures that don't support new kwargs
            # (e.g. unit-test stubs).
            docs = retrieval_fn(subproblem["text"], 2)

        retrieved[node_id] = docs
        answer = solver_fn(subproblem, docs, answers)

        # A4 — contradiction check against prior answers BEFORE inserting
        # the current node's answer, so the node doesn't compare against itself.
        prior_answers_snapshot = dict(answers)  # snapshot before mutation
        answers[node_id] = answer
        contradiction = check_contradiction(answer, prior_answers_snapshot)
        if contradiction:
            any_contradiction = True

        yield {
            "type": "node",
            "node_id": node_id,
            "subproblem": subproblem,
            "docs": docs,
            "answer": answer,
            "contradiction": contradiction,  # A4
        }

    synthesis_prompt = (
        f"Question to answer: {question}\n\n"
        "Facts gathered (use ONLY these — do not add outside knowledge):\n"
        + "\n".join(f"- {answers[node_id]}" for node_id in order)
        + "\n\n"
        "Now answer the question in as few words as possible.\n"
        "Rules:\n"
        "- For yes/no questions → reply ONLY 'yes' or 'no'\n"
        "- For 'what state' → give the state name only\n"
        "- For 'what theater' → give the theater name only\n"
        "- For 'why/what made' → give the reason/event only\n"
        "- For 'what nationality' → adjective form (e.g. 'Bulgarian' not 'Bulgaria')\n"
        "- For 'which genus/county/duchy' → that entity name only\n"
        "- For numbers/laps/dates → exact value with unit (e.g. '25 laps')\n"
        "- 1–5 words maximum. No explanations. No full sentences."
    )
    final_answer = call_llm(
        synthesis_prompt,
        system=(
            "You extract the exact answer from the facts provided. "
            "NEVER use external knowledge or training memory. "
            "ONLY read from the facts listed above. "
            "Output 1-5 words. No sentences."
        ),
        temperature=0.0,
    )
    yield {
        "type": "final",
        "order": order,
        "retrieved_docs": retrieved,
        "intermediate_answers": answers,
        "final_answer": final_answer,
        "contradictions_found": any_contradiction,  # A4
    }


# ---------------------------------------------------------------------------
# Non-streaming convenience wrapper
# ---------------------------------------------------------------------------

def resolve_graph(
    graph: nx.DiGraph,
    retrieval_fn: Callable[..., list],
    solver_fn: Callable[[dict, list, dict[str, str]], str] = resolve_node,
    question: str = "",
) -> dict[str, Any]:
    """Non-streaming convenience wrapper kept for callers that just want the
    final result (e.g. cached-run rendering, tests)."""
    result: dict[str, Any] | None = None
    for event in resolve_graph_stream(graph, retrieval_fn, solver_fn, question=question):
        if event["type"] == "final":
            result = {
                "order": event["order"],
                "retrieved_docs": event["retrieved_docs"],
                "intermediate_answers": event["intermediate_answers"],
                "final_answer": event["final_answer"],
                "contradictions_found": event["contradictions_found"],  # A4
            }
    assert result is not None
    return result

