"""One-time script: build a 25-example HotpotQA subset for the RL-LAG demo.

Produces:
  corpus/hotpot_corpus.jsonl    -- deduplicated paragraphs for FAISS indexing
  corpus/hotpot_questions.jsonl -- sampled questions for the Streamlit dropdown

Usage:
  python scripts/build_hotpot_subset.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = ROOT / "corpus" / "hotpot_corpus.jsonl"
QUESTIONS_OUT = ROOT / "corpus" / "hotpot_questions.jsonl"
SUBSET_SIZE = 25
RNG_SEED = 42


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading HotpotQA validation/distractor split (one-time download ~200 MB)...")
    # Dataset was moved to the hotpotqa namespace on Hugging Face Hub
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    # ── Inspect schema so field names are confirmed, not assumed ─────────────
    first = dataset[0]
    print("\n-- First record keys ------------------------------------------")
    for k, v in first.items():
        display = str(v)[:120] + "..." if len(str(v)) > 120 else str(v)
        print(f"  {k!r}: {display}")
    print("---------------------------------------------------------------\n")

    # Confirmed field names from HotpotQA distractor schema:
    #   question, answer, id, type, level, supporting_facts, context
    # context is a dict with keys 'title' (list[str]) and 'sentences' (list[list[str]])

    # ── Filter ───────────────────────────────────────────────────────────────
    # Keep hard/medium bridge or comparison questions (multi-hop DAG variety)
    def keep(example: dict) -> bool:
        level_ok = example.get("level", "") in ("hard", "medium")
        type_ok = example.get("type", "") in ("bridge", "comparison")
        return level_ok and type_ok

    filtered = [ex for ex in dataset if keep(ex)]
    print(f"Filtered to {len(filtered)} hard/medium bridge+comparison examples "
          f"(from {len(dataset)} total).")

    # ── Sample ───────────────────────────────────────────────────────────────
    random.seed(RNG_SEED)
    # Ensure we get a mix of bridge and comparison
    bridge_pool = [ex for ex in filtered if ex.get("type") == "bridge"]
    comparison_pool = [ex for ex in filtered if ex.get("type") == "comparison"]

    n_comparison = min(8, len(comparison_pool))        # at least 8 comparison
    n_bridge = SUBSET_SIZE - n_comparison              # rest are bridge

    sampled_bridge = random.sample(bridge_pool, min(n_bridge, len(bridge_pool)))
    sampled_comparison = random.sample(comparison_pool, n_comparison)
    sampled = sampled_bridge + sampled_comparison
    random.shuffle(sampled)                            # mix them in the dropdown

    print(f"Sampled {len(sampled_bridge)} bridge + {len(sampled_comparison)} "
          f"comparison = {len(sampled)} examples.")

    # ── Extract paragraphs ───────────────────────────────────────────────────
    seen_texts: set[str] = set()
    paragraphs: list[dict] = []
    para_id = 0

    for ex in sampled:
        ctx = ex["context"]
        titles = ctx["title"]
        sentences_per_title = ctx["sentences"]
        for title, sents in zip(titles, sentences_per_title):
            text = " ".join(s.strip() for s in sents).strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            paragraphs.append({
                "id": f"para_{para_id:04d}",
                "title": title,
                "text": text,
            })
            para_id += 1

    print(f"Extracted {len(paragraphs)} deduplicated paragraphs.")

    # ── Extract questions ────────────────────────────────────────────────────
    questions: list[dict] = []
    for i, ex in enumerate(sampled):
        ctx = ex["context"]
        supporting_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))
        questions.append({
            "id": f"q_{i+1:04d}",
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex["type"],
            "level": ex["level"],
            "supporting_titles": supporting_titles,
        })

    # ── Write outputs ────────────────────────────────────────────────────────
    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)

    with CORPUS_OUT.open("w", encoding="utf-8") as f:
        for para in paragraphs:
            f.write(json.dumps(para, ensure_ascii=False) + "\n")
    print(f"Written: {CORPUS_OUT}")

    with QUESTIONS_OUT.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Written: {QUESTIONS_OUT}")

    # ── Sanity checks ────────────────────────────────────────────────────────
    texts = [p["text"] for p in paragraphs]
    assert len(texts) == len(set(texts)), "Duplicate paragraphs detected!"
    bridge_qs = [q for q in questions if q["type"] == "bridge"]
    comp_qs = [q for q in questions if q["type"] == "comparison"]
    assert bridge_qs, "No bridge questions in subset!"
    assert comp_qs, "No comparison questions in subset!"

    print(f"\n✅ Done. {len(paragraphs)} paragraphs | {len(questions)} questions "
          f"({len(bridge_qs)} bridge, {len(comp_qs)} comparison)")
    print("\nSample questions:")
    for q in questions[:5]:
        print(f"  [{q['type']}/{q['level']}] {q['question']}")


if __name__ == "__main__":
    main()
