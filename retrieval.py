"""Small closed-world FAISS retriever — backed by HotpotQA corpus JSONL.

Upgrades (Track A):
  A1 — optional prior_context param enriches the embedding query so dependent
       nodes retrieve passages informed by earlier resolved answers.
  A3 — subproblem_type param:
         temporal    → k+1 (timelines benefit from extra supporting passages)
         comparative → min(k,2) (two-entity questions need precision over breadth)
         others      → caller-supplied k unchanged
  A5 — cosine-similarity dedup at threshold 0.92 removes near-duplicate
       passages from the result set before returning.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
# Primary corpus: HotpotQA paragraph subset (produced by scripts/build_hotpot_subset.py)
CORPUS_PATH = ROOT / "corpus" / "hotpot_corpus.jsonl"
# Fallback corpus: original hand-written knowledge base
FALLBACK_CORPUS_PATH = ROOT / "corpus" / "knowledge.txt"
VECTOR_DIR = ROOT / "vector_db"
INDEX_PATH = VECTOR_DIR / "knowledge.faiss"
META_PATH = VECTOR_DIR / "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# A5: similarity threshold above which two passages are considered near-duplicate.
DEDUP_THRESHOLD = 0.92



class LocalRetriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.passages, self.titles = self._read_corpus()
        self.index = self._load_or_build()

    def _read_corpus(self) -> tuple[list[str], list[str]]:
        """Load paragraphs from hotpot_corpus.jsonl; fall back to knowledge.txt."""
        if CORPUS_PATH.exists():
            records = [
                json.loads(line)
                for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            passages = [r["text"] for r in records]
            titles = [r.get("title", "") for r in records]
        elif FALLBACK_CORPUS_PATH.exists():
            lines = [
                line.strip()
                for line in FALLBACK_CORPUS_PATH.read_text(encoding="utf-8").splitlines()
            ]
            passages = [l for l in lines if l and not l.startswith("#")]
            titles = [""] * len(passages)
        else:
            raise RuntimeError(
                "No corpus found. Run: python scripts/build_hotpot_subset.py"
            )

        if not passages:
            raise RuntimeError("The local corpus is empty.")
        return passages, titles

    def _load_or_build(self):
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        corpus_signature = {"passages": self.passages, "model": MODEL_NAME}
        if INDEX_PATH.exists() and META_PATH.exists():
            try:
                metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
                if metadata == corpus_signature:
                    return faiss.read_index(str(INDEX_PATH))
            except (json.JSONDecodeError, OSError):
                pass

        embeddings = self.model.encode(
            self.passages,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(INDEX_PATH))
        META_PATH.write_text(json.dumps(corpus_signature, indent=2), encoding="utf-8")
        return index

    def retrieve(
        self,
        query: str,
        k: int = 3,
        prior_context: str = "",
        subproblem_type: str = "factual",
    ) -> list[dict]:
        """Return top-k deduplicated passages with a 0–1 cosine similarity score and title.

        A1: if prior_context is provided, it is prepended to the query string
            before encoding so the embedding reflects the accumulated reasoning state.
        A3: temporal/comparative subproblem types cap k at 2 to prefer concise,
            precise evidence rather than a wider (noisier) pool.
        A5: near-duplicate passages (cosine ≥ 0.92) are removed before returning.
        """
        # A1 — enrich query with prior context for dependent nodes.
        effective_query = (
            f"{prior_context.strip()} {query.strip()}".strip()
            if prior_context
            else query
        )

        # A3 — type-aware k adjustment per roadmap spec:
        #   temporal    → +1 (timelines often need an extra supporting passage)
        #   comparative → cap at 2 (two-entity questions need precision over breadth)
        #   others      → caller-supplied k unchanged
        if subproblem_type == "temporal":
            effective_k = k + 1
        elif subproblem_type == "comparative":
            effective_k = min(k, 2)
        else:
            effective_k = k

        query_vector = self.model.encode(
            [effective_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        # Retrieve more candidates than needed so dedup still returns effective_k results.
        fetch_k = min(effective_k * 3, len(self.passages))
        scores, indices = self.index.search(np.asarray(query_vector), fetch_k)

        candidates = []
        for score, i in zip(scores[0], indices[0]):
            if 0 <= i < len(self.passages):
                candidates.append({
                    "text": self.passages[i],
                    "title": self.titles[i],
                    "score": max(0.0, min(1.0, float(score))),
                })

        # A5 — cosine-similarity dedup.
        deduplicated = _dedup_passages(candidates, self.model, DEDUP_THRESHOLD)
        return deduplicated[:effective_k]


def _dedup_passages(
    candidates: list[dict],
    model: SentenceTransformer,
    threshold: float,
) -> list[dict]:
    """Remove near-duplicate passages using pairwise cosine similarity.

    Passages are already sorted by descending score from FAISS. We iterate in
    that order and skip any candidate that is cosine-similar (≥ threshold) to an
    already-accepted passage.
    """
    if len(candidates) <= 1:
        return candidates

    texts = [c["text"] for c in candidates]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")  # shape: (n, dim)

    accepted_indices: list[int] = []
    for i in range(len(candidates)):
        is_dup = False
        for j in accepted_indices:
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            accepted_indices.append(i)

    return [candidates[i] for i in accepted_indices]


_retriever: LocalRetriever | None = None


def get_retriever() -> LocalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LocalRetriever()
    return _retriever


def retrieve(
    query: str,
    k: int = 3,
    prior_context: str = "",
    subproblem_type: str = "factual",
) -> list[dict]:
    """Return list of {"text": str, "title": str, "score": float} dicts, top-k by similarity.

    A1: prior_context — prior node answers to condition the embedding query.
    A3: subproblem_type — 'temporal'/'comparative' use a reduced k; others unchanged.
    A5: near-duplicate passages are removed before returning.
    """
    return get_retriever().retrieve(
        query,
        k=k,
        prior_context=prior_context,
        subproblem_type=subproblem_type,
    )
