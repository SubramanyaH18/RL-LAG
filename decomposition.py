"""Question decomposition for the RL-LAG architectural prototype."""
from __future__ import annotations

import json
import re
from typing import Any

from llm_client import call_llm

MAX_SUBPROBLEMS = 6

SYSTEM = """\
You decompose multi-hop questions into a minimal least-to-most plan.

DEFINITIONS
-----------
A "multi-hop" question requires resolving an intermediate entity or fact before
the final fact can be answered.  Every such question MUST be split into at least
two subproblems where the later one depends on the earlier one.

Collapsing a multi-hop question into a single subproblem that repeats the
original question verbatim is INVALID and will be rejected automatically.
If you do that, your answer will be discarded and you will be asked again.

DEPENDENCY RULES
----------------
- depends_on must list the IDs of earlier subproblems (in the same response)
  that this subproblem needs.
- IDs must be spelled exactly as you defined them (e.g. "q1", "q2", …).
- References to IDs that have not been defined yet are illegal.

ALLOWED TYPES: factual | relational | comparative | temporal

───────────────────────────── FEW-SHOT EXAMPLES ─────────────────────────────

## Example 1 — Bridge question (multi-hop, 2 nodes)
Question: "What is the nationality of the director of Inception?"
CORRECT decomposition:
[
  {"id":"q1","text":"Who directed Inception?","type":"factual","depends_on":[]},
  {"id":"q2","text":"What is the nationality of Christopher Nolan?","type":"relational","depends_on":["q1"]}
]
WRONG (single node that echoes the question — REJECTED):
[{"id":"q1","text":"What is the nationality of the director of Inception?","type":"factual","depends_on":[]}]

## Example 2 — Comparison question (multi-hop, 3 nodes)
Question: "Which film has a higher IMDb rating, Inception or Interstellar?"
CORRECT decomposition:
[
  {"id":"q1","text":"What is the IMDb rating of Inception?","type":"factual","depends_on":[]},
  {"id":"q2","text":"What is the IMDb rating of Interstellar?","type":"factual","depends_on":[]},
  {"id":"q3","text":"Which film has a higher IMDb rating: Inception or Interstellar?","type":"comparative","depends_on":["q1","q2"]}
]

## Example 3 — Genuinely single-hop (1 node is correct)
Question: "What is the capital of France?"
CORRECT decomposition:
[{"id":"q1","text":"What is the capital of France?","type":"factual","depends_on":[]}]
(No intermediate entity lookup is needed — a single node is fine here.)

──────────────────────────────────────────────────────────────────────────────
Return ONLY a valid JSON array — no markdown fences, no prose before or after.\
"""


def _extract_json(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")
    return parsed


def _normalize(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise raw LLM output into clean subproblem dicts.

    Improvements over the original:
    - Resolves depends_on references case-insensitively and whitespace-trimmed,
      so "Q1" resolves to "q1" without silently dropping the edge.
    - Prints an explicit warning when a reference cannot be resolved at all.
    """
    valid_types = {"factual", "relational", "comparative", "temporal"}
    result: list[dict[str, Any]] = []
    seen_canonical: dict[str, str] = {}  # lowercased_id -> canonical_id

    for index, raw in enumerate(items[:MAX_SUBPROBLEMS], start=1):
        node_id = str(raw.get("id") or f"q{index}")
        if node_id.lower() in seen_canonical:
            node_id = f"q{index}"
        text = str(raw.get("text") or raw.get("question") or "").strip()
        if not text:
            continue

        deps_raw = raw.get("depends_on") or []
        if isinstance(deps_raw, str):
            deps_raw = [deps_raw]

        resolved_deps: list[str] = []
        for dep in deps_raw:
            dep_key = str(dep).strip().lower()
            if dep_key in seen_canonical:
                resolved_deps.append(seen_canonical[dep_key])
            else:
                known = list(seen_canonical.values())
                print(
                    f"[decomposition] warning: node '{node_id}' has unresolved "
                    f"dependency '{dep}' (known ids so far: {known}); dropping edge."
                )

        kind = str(raw.get("type") or "factual").lower()
        if kind not in valid_types:
            kind = "factual"

        result.append({"id": node_id, "text": text, "type": kind, "depends_on": resolved_deps})
        seen_canonical[node_id.lower()] = node_id

    if len(items) > MAX_SUBPROBLEMS:
        print(f"[decomposition] warning: truncated to {MAX_SUBPROBLEMS} subproblems")
    if not result:
        raise ValueError("No valid subproblems returned")
    return result


def _fallback_lines(text: str, question: str) -> list[dict[str, Any]]:
    lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)]|q\d+[:.)-]?)\s*", "", line).strip()
        for line in text.splitlines()
    ]
    lines = [line for line in lines if len(line) > 5][:MAX_SUBPROBLEMS]
    if not lines:
        lines = [question]
    return [
        {
            "id": f"q{i}",
            "text": line,
            "type": "factual",
            "depends_on": [] if i == 1 else [f"q{i-1}"],
        }
        for i, line in enumerate(lines, start=1)
    ]


def _is_noop_decomposition(subproblems: list[dict[str, Any]], question: str) -> bool:
    """Return True iff the decomposition is a degenerate single-node echo.

    A 'no-op' decomposition has exactly one subproblem whose text, after
    lowercasing and stripping all non-alphanumeric characters, is identical
    to the same normalisation of the original question.
    """
    if len(subproblems) != 1:
        return False
    _strip = re.compile(r"[^a-z0-9]")
    norm_q = _strip.sub("", question.lower())
    norm_s = _strip.sub("", subproblems[0]["text"].lower())
    return norm_q == norm_s


def decompose_query(question: str) -> list[dict[str, Any]]:
    base_prompt = f"""Question: {question}

Identify the minimal atomic subproblems needed to answer it. Use at most {MAX_SUBPROBLEMS}.
Each item must contain: id, text, type, depends_on.
Allowed types: factual, relational, comparative, temporal.
Return ONLY valid JSON in this exact shape:
[{{"id":"q1","text":"...","type":"factual","depends_on":[]}}]"""

    # ── First attempt ──────────────────────────────────────────────────────────
    first = call_llm(base_prompt, system=SYSTEM, temperature=0.1)
    try:
        parsed_first = _normalize(_extract_json(first))
        noop = _is_noop_decomposition(parsed_first, question)
        if not noop:
            return parsed_first
        # Fall through to retry when it's a no-op
    except (ValueError, json.JSONDecodeError, TypeError):
        noop = True  # parse failure also triggers retry

    # ── Retry with an explicit rejection message ───────────────────────────────
    retry_prompt = (
        base_prompt
        + "\n\nYour previous output was REJECTED because you returned a single subproblem "
        "that merely repeats the original question verbatim. That is not a decomposition.\n"
        "Instead:\n"
        "  1. Identify the intermediate entity or fact that must be looked up first.\n"
        "  2. Make that the first subproblem (depends_on: []).\n"
        "  3. Make the final fact about that entity the second subproblem "
        "(depends_on: [\"q1\"]).\n"
        "Return ONLY the JSON array, no markdown or explanation."
    )
    second = call_llm(retry_prompt, system=SYSTEM, temperature=0.0)
    try:
        parsed_second = _normalize(_extract_json(second))
        if _is_noop_decomposition(parsed_second, question):
            print(
                "[decomposition] warning: model returned a no-op decomposition after retry. "
                "Proceeding with a single-node graph. Any correct final answer in this case "
                "did NOT come from verified multi-hop reasoning."
            )
        return parsed_second
    except (ValueError, json.JSONDecodeError, TypeError):
        print(
            "[decomposition] warning: model returned a no-op decomposition after retry "
            "(parse also failed — using line-based fallback). "
            "Proceeding with a single-node graph. Any correct final answer in this case "
            "did NOT come from verified multi-hop reasoning."
        )
        return _fallback_lines(second, question)
