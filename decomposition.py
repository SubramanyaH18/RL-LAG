"""Question decomposition for the RL-LAG architectural prototype."""
from __future__ import annotations

import json
import re
from typing import Any

from llm_client import call_llm

MAX_SUBPROBLEMS = 6

SYSTEM = """You decompose multi-hop questions into a minimal least-to-most plan.
Return atomic subproblems only. Dependencies must reference earlier IDs."""


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
    valid_types = {"factual", "relational", "comparative", "temporal"}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(items[:MAX_SUBPROBLEMS], start=1):
        node_id = str(raw.get("id") or f"q{index}")
        if node_id in seen:
            node_id = f"q{index}"
        text = str(raw.get("text") or raw.get("question") or "").strip()
        if not text:
            continue
        deps = raw.get("depends_on") or []
        if isinstance(deps, str):
            deps = [deps]
        deps = [str(dep) for dep in deps if str(dep) in seen]
        kind = str(raw.get("type") or "factual").lower()
        if kind not in valid_types:
            kind = "factual"
        result.append({"id": node_id, "text": text, "type": kind, "depends_on": deps})
        seen.add(node_id)
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


def decompose_query(question: str) -> list[dict[str, Any]]:
    prompt = f"""Question: {question}

Identify the minimal atomic subproblems needed to answer it. Use at most {MAX_SUBPROBLEMS}.
Each item must contain: id, text, type, depends_on.
Allowed types: factual, relational, comparative, temporal.
Return ONLY valid JSON in this exact shape:
[{{"id":"q1","text":"...","type":"factual","depends_on":[]}}]"""
    first = call_llm(prompt, system=SYSTEM, temperature=0.1)
    try:
        return _normalize(_extract_json(first))
    except (ValueError, json.JSONDecodeError, TypeError):
        retry_prompt = prompt + "\nYour previous output was invalid. Return ONLY the JSON array, with no markdown or explanation."
        second = call_llm(retry_prompt, system=SYSTEM, temperature=0.0)
        try:
            return _normalize(_extract_json(second))
        except (ValueError, json.JSONDecodeError, TypeError):
            print("[decomposition] warning: using line-based fallback")
            return _fallback_lines(second, question)
