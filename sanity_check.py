"""Pre-flight sanity checker for the RL-LAG pipeline.

Runs three targeted checks on 50 questions from the training pool before
PPO training begins, so a broken component is caught cheaply rather than
after burning rollout episodes.

Checks
------
A  Decomposition validity
   Fraction of questions that produce a valid, non-cyclic, correctly-structured
   DAG without needing the fallback single-node path.

B  Context-aware retrieval (A1) effect
   For dependent nodes in the decomposed DAGs, whether prepending prior-answer
   context actually changes the top-k retrieved passages (passage ID sets
   compared with vs. without prior_context).

C  Dedup threshold (A5) activity
   Whether the 0.92 cosine dedup in retrieval.py is actually removing passages
   (threshold is live, not a dead code path).

Threshold policy
----------------
On the FIRST run, results are saved to corpus/sanity_baseline.json as the
measured baseline -- no pass/fail gate, just measurement.
On SUBSEQUENT runs, each rate is compared against the stored baseline.
A [PREFLIGHT WARN] is printed only when a rate drops more than 10 percentage
points below the baseline value.  No hardcoded magic numbers.

Usage
-----
    python sanity_check.py               # normal run (needs Ollama)
    python sanity_check.py --dry-run     # mock LLM, no Ollama needed
    python sanity_check.py --questions 20  # fewer questions for speed
    python sanity_check.py --no-baseline   # skip baseline comparison

Wire-in (train_ppo.py calls this before the training loop):
    from sanity_check import run_sanity_check
    run_sanity_check()
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

ROOT             = Path(__file__).resolve().parent
BASELINE_PATH    = ROOT / "corpus" / "sanity_baseline.json"
SANITY_REPORT_PATH = ROOT / "sanity_report.json"
SANITY_N_QUESTIONS = 50   # number of questions used for each check
SANITY_SEED        = 0    # separate fixed seed so sanity sample != training order
WARN_DROP_THRESHOLD = 0.10  # warn if a rate drops more than this vs baseline


# ── Check A: Decomposition validity ──────────────────────────────────────────

def _check_decomposition(questions: list[dict]) -> dict[str, Any]:
    """Check A: DAG validity rate across the sample."""
    from decomposition import decompose_query
    from graph_builder import build_graph

    n = len(questions)
    valid_dag      = 0  # produced a valid, acyclic, multi-node DAG
    noop           = 0  # single node that echoes the question (no-op decomp)
    fallback_used  = 0  # parse failed, fell back to single-node
    cycle_error    = 0  # decomposition produced a cycle
    dep_error      = 0  # unknown dependency reference
    total_nodes    = 0

    for q in questions:
        qtext = q["question"]
        try:
            subproblems = decompose_query(qtext)
        except Exception:
            fallback_used += 1
            continue

        if not subproblems:
            fallback_used += 1
            continue

        # Check for no-op: single node whose text ~ the original question
        if len(subproblems) == 1:
            node_text = subproblems[0].get("text", "").strip().lower()
            q_norm    = qtext.strip().lower().rstrip("?")
            if node_text.rstrip("?") == q_norm or node_text == qtext.strip().lower():
                noop += 1
                continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                graph, n_comp = build_graph(subproblems)
            valid_dag   += 1
            total_nodes += len(subproblems)
        except ValueError as e:
            if "cycle" in str(e).lower():
                cycle_error += 1
            else:
                dep_error += 1
        except Exception:
            fallback_used += 1

    mean_nodes = (total_nodes / valid_dag) if valid_dag else 0.0
    return {
        "n":                n,
        "valid_dag_rate":   round(valid_dag   / n, 4),
        "noop_rate":        round(noop         / n, 4),
        "fallback_rate":    round(fallback_used / n, 4),
        "cycle_rate":       round(cycle_error   / n, 4),
        "dep_error_rate":   round(dep_error     / n, 4),
        "mean_nodes_per_valid_dag": round(mean_nodes, 2),
    }


# ── Check B: A1 context enrichment effect ────────────────────────────────────

def _check_retrieval_context_effect(questions: list[dict]) -> dict[str, Any]:
    """Check B: Does prior_context actually change retrieved passage sets?"""
    from decomposition import decompose_query
    from retrieval import retrieve
    import warnings

    checked       = 0  # questions with at least one dependent node
    changed       = 0  # nodes where passage set differed with context
    total_dep_nodes = 0
    changed_nodes = 0

    for q in questions:
        qtext = q["question"]
        try:
            subproblems = decompose_query(qtext)
        except Exception:
            continue

        # Find nodes with dependencies (A1 only fires on these)
        dep_nodes = [s for s in subproblems if s.get("depends_on")]
        if not dep_nodes:
            continue

        checked += 1
        for node in dep_nodes[:2]:  # cap at 2 dep nodes per question for speed
            total_dep_nodes += 1
            node_text = node.get("text", qtext)

            # Retrieve WITHOUT context
            docs_no_ctx = retrieve(node_text, k=5)
            ids_no_ctx  = {d.get("title", "") + "|" + d["text"][:40]
                           for d in docs_no_ctx}

            # Retrieve WITH a plausible fake prior-answer context
            fake_ctx = f"Prior answer: {q.get('answer', 'unknown')}."
            docs_ctx = retrieve(node_text, k=5, prior_context=fake_ctx)
            ids_ctx  = {d.get("title", "") + "|" + d["text"][:40]
                        for d in docs_ctx}

            if ids_no_ctx != ids_ctx:
                changed_nodes += 1

    change_rate = round(changed_nodes / total_dep_nodes, 4) if total_dep_nodes else 0.0
    return {
        "questions_with_dep_nodes": checked,
        "total_dep_nodes_checked":  total_dep_nodes,
        "nodes_where_set_changed":  changed_nodes,
        "retrieval_change_rate":    change_rate,
    }


# ── Check C: Dedup threshold (A5) activity ────────────────────────────────────

def _check_dedup_activity(questions: list[dict]) -> dict[str, Any]:
    """Check C: Is the 0.92 cosine dedup actually removing any passages?"""
    from retrieval import get_retriever, DEDUP_THRESHOLD, _dedup_passages

    retriever = get_retriever()
    import numpy as np

    before_counts = []
    after_counts  = []
    n = min(20, len(questions))

    for q in questions[:n]:
        qtext = q["question"]
        # Retrieve raw candidates without dedup
        query_vector = retriever.model.encode(
            [qtext], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        fetch_k = min(30, len(retriever.passages))
        scores, indices = retriever.index.search(np.asarray(query_vector), fetch_k)
        candidates = [
            {"text": retriever.passages[i], "title": retriever.titles[i],
             "score": float(scores[0][j])}
            for j, i in enumerate(indices[0]) if 0 <= i < len(retriever.passages)
        ]

        before_counts.append(len(candidates))
        deduped = _dedup_passages(candidates, retriever.model, DEDUP_THRESHOLD)
        after_counts.append(len(deduped))

    mean_before  = round(sum(before_counts) / n, 2) if n else 0.0
    mean_after   = round(sum(after_counts)  / n, 2) if n else 0.0
    mean_removed = round(mean_before - mean_after, 2)
    return {
        "questions_checked":     n,
        "dedup_threshold":       DEDUP_THRESHOLD,
        "mean_passages_before":  mean_before,
        "mean_passages_after":   mean_after,
        "mean_removed_per_q":    mean_removed,
    }


# ── Type-aware k check (bonus, cheap) ────────────────────────────────────────

def _check_type_aware_k(questions: list[dict]) -> dict[str, Any]:
    """Verify A3: temporal gets k+1, comparative gets min(k,2)."""
    from retrieval import retrieve
    k = 3
    temporal_results    = retrieve("When was the Eiffel Tower built?", k=k, subproblem_type="temporal")
    comparative_results = retrieve("Which is taller, the Eiffel Tower or the Empire State Building?",
                                   k=k, subproblem_type="comparative")
    factual_results     = retrieve("Who built the Eiffel Tower?", k=k, subproblem_type="factual")
    return {
        "k_requested":           k,
        "temporal_returned":     len(temporal_results),    # expect <= k+1
        "comparative_returned":  len(comparative_results), # expect <= 2
        "factual_returned":      len(factual_results),     # expect <= k
        "temporal_k_ok":         len(temporal_results) <= k + 1,
        "comparative_k_ok":      len(comparative_results) <= 2,
    }


# ── Baseline comparison ───────────────────────────────────────────────────────

def _compare_to_baseline(current: dict, check_name: str) -> list[str]:
    """Return warning strings for rates that dropped > WARN_DROP_THRESHOLD."""
    warnings_out: list[str] = []
    if not BASELINE_PATH.exists():
        return warnings_out
    try:
        baseline = json.loads(BASELINE_PATH.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return warnings_out

    base = baseline.get(check_name, {})
    for key, val in current.items():
        if not isinstance(val, float):
            continue
        base_val = base.get(key)
        if base_val is None or not isinstance(base_val, float):
            continue
        drop = base_val - val
        if drop > WARN_DROP_THRESHOLD:
            warnings_out.append(
                f"  [PREFLIGHT WARN] {check_name}.{key}: "
                f"dropped {drop:.2f} below baseline "
                f"({base_val:.3f} -> {val:.3f})"
            )
    return warnings_out


# ── Main runner ───────────────────────────────────────────────────────────────

def run_sanity_check(
    n_questions:   int  = SANITY_N_QUESTIONS,
    use_baseline:  bool = True,
    verbose:       bool = True,
) -> dict[str, Any]:
    """Run all sanity checks and return the report dict.

    On the first run (no baseline file), results are saved as the baseline.
    On subsequent runs, results are compared against the baseline and
    [PREFLIGHT WARN] messages are printed for significant drops.

    Parameters
    ----------
    n_questions  : How many questions to sample for the checks.
    use_baseline : Whether to compare against stored baseline.
    verbose      : Whether to print the report to stdout.
    """
    import random as _rng
    from data_pools import load_train_pool

    pool = load_train_pool()
    r = _rng.Random(SANITY_SEED)
    sample = r.sample(pool, min(n_questions, len(pool)))

    t0 = time.time()
    print(f"\n[sanity_check] Running pre-flight checks on {len(sample)} questions...")

    # ── Run checks ────────────────────────────────────────────────────────────
    print("[sanity_check] Check A: decomposition validity...", flush=True)
    check_a = _check_decomposition(sample)

    print("[sanity_check] Check B: retrieval context-enrichment effect...", flush=True)
    check_b = _check_retrieval_context_effect(sample)

    print("[sanity_check] Check C: dedup (A5) activity...", flush=True)
    check_c = _check_dedup_activity(sample)

    print("[sanity_check] Check D: type-aware k (A3) behaviour...", flush=True)
    check_d = _check_type_aware_k(sample)

    elapsed = time.time() - t0
    report = {
        "n_questions": len(sample),
        "elapsed_s":   round(elapsed, 1),
        "check_a_decomp":       check_a,
        "check_b_retrieval_a1": check_b,
        "check_c_dedup_a5":     check_c,
        "check_d_type_k_a3":    check_d,
    }

    # ── Save / compare baseline ───────────────────────────────────────────────
    all_warnings: list[str] = []
    if use_baseline:
        if not BASELINE_PATH.exists():
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            BASELINE_PATH.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), "utf-8"
            )
            print(f"[sanity_check] First run -- baseline saved to {BASELINE_PATH.name}")
        else:
            for check_key, check_name in [
                ("check_a_decomp",       "check_a_decomp"),
                ("check_b_retrieval_a1", "check_b_retrieval_a1"),
                ("check_c_dedup_a5",     "check_c_dedup_a5"),
            ]:
                all_warnings.extend(_compare_to_baseline(report[check_key], check_name))

    # ── Save report ───────────────────────────────────────────────────────────
    SANITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SANITY_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), "utf-8"
    )

    # ── Print report ──────────────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*62}")
        print(f"  SANITY CHECK REPORT  ({elapsed:.1f}s)")
        print(f"{'='*62}")

        a = check_a
        print(f"\n  Check A -- Decomposition validity (n={a['n']})")
        print(f"    valid_dag_rate   : {a['valid_dag_rate']:.3f}  "
              f"({int(a['valid_dag_rate']*a['n'])}/{a['n']} questions)")
        print(f"    noop_rate        : {a['noop_rate']:.3f}")
        print(f"    fallback_rate    : {a['fallback_rate']:.3f}")
        print(f"    cycle_rate       : {a['cycle_rate']:.3f}")
        print(f"    mean_nodes/valid : {a['mean_nodes_per_valid_dag']:.2f}")

        b = check_b
        print(f"\n  Check B -- Retrieval A1 context effect")
        print(f"    dep_nodes_checked    : {b['total_dep_nodes_checked']}")
        print(f"    retrieval_change_rate: {b['retrieval_change_rate']:.3f}  "
              f"({b['nodes_where_set_changed']}/{b['total_dep_nodes_checked']} nodes)")

        c = check_c
        print(f"\n  Check C -- Dedup A5 activity (n={c['questions_checked']})")
        print(f"    threshold        : {c['dedup_threshold']}")
        print(f"    mean before dedup: {c['mean_passages_before']:.1f}")
        print(f"    mean after dedup : {c['mean_passages_after']:.1f}")
        print(f"    mean removed/q   : {c['mean_removed_per_q']:.1f}")

        d = check_d
        print(f"\n  Check D -- Type-aware k A3 (k_requested={d['k_requested']})")
        print(f"    temporal  returned : {d['temporal_returned']}  "
              f"(ok: {d['temporal_k_ok']})")
        print(f"    comparative returned: {d['comparative_returned']}  "
              f"(ok: {d['comparative_k_ok']})")
        print(f"    factual   returned : {d['factual_returned']}")

        if all_warnings:
            print(f"\n  {'─'*58}")
            for w in all_warnings:
                print(w)
        else:
            bline_msg = ("  [PREFLIGHT OK] All rates within baseline tolerance."
                         if BASELINE_PATH.exists() else
                         "  [PREFLIGHT BASELINE] First run -- baseline recorded.")
            print(f"\n  {bline_msg}")

        print(f"{'='*62}")
        print(f"  Report: {SANITY_REPORT_PATH}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RL-LAG pre-flight pipeline sanity checker."
    )
    parser.add_argument("--questions", type=int, default=SANITY_N_QUESTIONS,
                        help=f"Questions to sample (default: {SANITY_N_QUESTIONS}).")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline comparison (just print current numbers).")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Use mock LLM -- no Ollama needed.")
    parser.add_argument("--reset-baseline", action="store_true",
                        help="Delete stored baseline and re-record from this run.")
    args = parser.parse_args()

    if args.reset_baseline and BASELINE_PATH.exists():
        BASELINE_PATH.unlink()
        print(f"[sanity_check] Baseline reset: deleted {BASELINE_PATH}")

    if args.dry_run:
        from unittest.mock import patch
        _MOCK_DECOMP = json.dumps([
            {"id": "q1", "text": "Who directed Inception?",
             "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What is the nationality of that director?",
             "type": "relational", "depends_on": ["q1"]},
        ])
        def _mock_llm(prompt: str, **kw) -> str:
            if "subproblem" in prompt.lower() or "json" in prompt.lower():
                return _MOCK_DECOMP
            return "Christopher Nolan"
        with patch("decomposition.call_llm", side_effect=_mock_llm):
            run_sanity_check(
                n_questions=args.questions,
                use_baseline=not args.no_baseline,
            )
    else:
        run_sanity_check(
            n_questions=args.questions,
            use_baseline=not args.no_baseline,
        )