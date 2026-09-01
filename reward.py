"""Rule-based reward with five explicit components for the RL-LAG prototype.

Upgrades (Track A → Track B integration):
  A6 — compute_reward() originally had four named components.
  B-fix — A 5th component, *correctness*, is added to close the reward/gold-answer
           gap.  Correctness is computed via eval.py's exact_match() and f1_score()
           against the gold answer for the current question.

  Current formula:
    R = w_c·correctness
      + α·retrieval_presence
      + β·(1 − token_cost)
      + γ·(1 − contradiction_penalty)
      − δ·hallucination_rate

  Correctness receives ~50 % of the total weight so that *getting the right answer*
  dominates the learning signal; the other four terms act as auxiliary shaping.

  All components are 0–1 normalised so the composite score is interpretable.

Honesty note
------------
The correctness signal is still a surface-level string comparison (EM + token-F1),
not a semantic equivalence check.  It will under-reward semantically correct answers
that use different phrasing (e.g. "NYC" vs "New York City").  This is a known
limitation inherited from the standard HotpotQA evaluation protocol.
"""
from __future__ import annotations

import re

from eval import exact_match, f1_score

TOKEN_BUDGET = 2048  # soft budget for token-cost normalisation (A6)

# Correctness dominates (~50 %) so the policies learn to produce correct answers;
# the other four terms are auxiliary shaping signals sharing the remaining 50 %.
_W_CORRECTNESS = 0.50   # answer correctness (EM + F1 blend against gold)
_ALPHA = 0.125          # retrieval presence
_BETA  = 0.125          # token efficiency
_GAMMA = 0.125          # logical consistency (no contradictions)
_DELTA = 0.125          # grounding / anti-hallucination


def _retrieval_presence_score(retrieved_docs_per_node: dict[str, list]) -> float:
    """Fraction of nodes that have at least one supporting passage."""
    if not retrieved_docs_per_node:
        return 0.0
    supported = sum(bool(docs) for docs in retrieved_docs_per_node.values())
    return supported / len(retrieved_docs_per_node)


def _token_cost_score(completion_tokens: int) -> float:
    """1 − (tokens_used / budget), clipped to [0, 1].

    Higher is better (lower token usage relative to the budget).
    """
    return max(0.0, min(1.0, 1.0 - completion_tokens / TOKEN_BUDGET))


def _hallucination_score(final_answer: str, retrieved_docs_per_node: dict[str, list]) -> float:
    """Sentence-level grounding check.

    For each sentence in final_answer, check whether it shares at least one
    content token (≥4 chars) with the body of retrieved passages.  Returns the
    fraction of sentences that are grounded.  A score of 1.0 means every
    sentence is supported by retrieved evidence.
    """
    # Collect all passage text.
    all_passage_text = " ".join(
        doc.get("text", "") if isinstance(doc, dict) else str(doc)
        for docs in retrieved_docs_per_node.values()
        for doc in docs
    ).lower()

    if not all_passage_text.strip():
        # No evidence retrieved at all — we can't judge grounding.
        return 0.5

    passage_tokens = set(re.findall(r"\b\w{4,}\b", all_passage_text))

    sentences = [s.strip() for s in re.split(r"[.!?]+", final_answer) if s.strip()]
    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        sentence_tokens = set(re.findall(r"\b\w{4,}\b", sentence.lower()))
        if sentence_tokens & passage_tokens:
            grounded += 1

    return grounded / len(sentences)


def _correctness_score(final_answer: str, gold_answer: str) -> float:
    """Blend of EM and token-F1 against the gold answer.

    Uses the existing eval.py helpers (A7).  A pure EM reward would be too
    sparse for a 3B frozen model to learn from; blending with F1 gives a
    smoother gradient signal even for partially-correct answers.

    Honesty note: this is still a surface-level string comparison, not a
    semantic equivalence check.  It will under-reward semantically correct
    answers that use different phrasing (e.g. "NYC" vs "New York City").
    """
    if not gold_answer:
        return 0.0
    em = exact_match(final_answer, gold_answer)
    f1 = f1_score(final_answer, gold_answer)
    return 0.5 * em + 0.5 * f1   # blended score, 0–1


def compute_reward(
    retrieved_docs_per_node: dict[str, list],
    final_answer: str,
    contradictions_found: bool,
    completion_tokens: int = 0,
    gold_answer: str = "",
) -> dict:
    """Compute a transparent five-component heuristic reward.

    Formula:
      R = w_c·correctness
        + α·retrieval_presence
        + β·(1 − token_cost)
        + γ·logical_consistency
        − δ·(1 − grounding)

    Correctness (EM + F1 blend) receives ~50 % of the weight so the policies
    are primarily rewarded for producing the right answer.  The other four
    terms act as auxiliary shaping signals.

    When gold_answer is empty or not provided, the correctness term contributes
    0.0 — this preserves backward compatibility for callers (app.py, eval.py)
    that do not have access to the gold answer.

    Honesty note: this is a rule-based reward, not a trained reward model.
    The correctness component is a surface-level string comparison (EM + F1),
    not a semantic equivalence check.  Components are individually logged so
    it is clear which term drives any change in the composite score.

    Parameters
    ----------
    retrieved_docs_per_node : dict[node_id -> list[doc]]
        Retrieved passages per node (from solver output).
    final_answer : str
        The synthesised final answer string.
    contradictions_found : bool
        True if any node's answer contradicted a prior answer (A4).
    completion_tokens : int
        Total completion tokens used in this pipeline run, from
        llm_client.get_usage_stats()["completion_tokens"] (A6).
    gold_answer : str
        The ground-truth answer for the current question.  When provided,
        enables the correctness component (EM + F1 blend).  When empty
        (default), correctness contributes 0.0.

    Returns
    -------
    dict with keys:
        score        — weighted composite (float, higher is better)
        components   — dict of the five named sub-scores
        explanation  — human-readable summary string
    """
    rp = _retrieval_presence_score(retrieved_docs_per_node)
    tc = _token_cost_score(completion_tokens)
    lc = 0.0 if contradictions_found else 1.0
    hal = _hallucination_score(final_answer, retrieved_docs_per_node)
    cor = _correctness_score(final_answer, gold_answer)

    score = round(
        _W_CORRECTNESS * cor
        + _ALPHA * rp
        + _BETA * tc
        + _GAMMA * lc
        - _DELTA * (1.0 - hal),
        4,
    )

    components = {
        "correctness": round(cor, 4),
        "retrieval_presence": round(rp, 4),
        "token_efficiency": round(tc, 4),
        "logical_consistency": round(lc, 4),
        "grounding": round(hal, 4),
    }

    supported = sum(bool(docs) for docs in retrieved_docs_per_node.values())
    total = len(retrieved_docs_per_node)
    explanation = (
        f"{supported}/{total} nodes had supporting evidence; "
        + ("contradictions were flagged. " if contradictions_found else "no contradictions flagged. ")
        + f"Token efficiency: {tc:.2f} ({completion_tokens}/{TOKEN_BUDGET} tokens used). "
        + f"Grounding score: {hal:.2f}."
    )
    if gold_answer:
        explanation += f" Correctness (EM+F1 blend): {cor:.2f}."
    if not final_answer.strip():
        explanation += " The final answer was empty."

    return {"score": score, "components": components, "explanation": explanation}
