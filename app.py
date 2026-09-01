"""Streamlit interface for the Ollama-powered RL-LAG prototype — HotpotQA edition.

Upgrades (Track A):
  A2 — shows an st.warning banner when the dependency graph has disconnected
       components (n_components > 1).
  A6 — render_reward_section() now displays all four reward components as
       individual st.metric() tiles; the section heading is updated from
       "Placeholder reward" to "Reward (4 components)".
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from graph_builder import TYPE_COLORS, build_graph, render_graph, render_graph_interactive
from llm_client import DEFAULT_MODEL, OLLAMA_HOST, get_usage_stats
from pipeline import run_pipeline_stream

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "demo_cache.json"
HOTPOT_QUESTIONS_PATH = ROOT / "corpus" / "hotpot_questions.jsonl"

st.set_page_config(page_title="RL-LAG Prototype — Ollama", layout="wide")
st.title("RL-LAG Architectural Prototype")
st.caption(
    "Local Ollama inference • FAISS retrieval • NetworkX dependency DAG • 4-component heuristic reward  "
    "| Corpus: HotpotQA validation/distractor subset (25 questions)"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Runtime")
    use_cached = st.toggle("Use cached demo run", value=True)
    st.write(f"**Model:** `{DEFAULT_MODEL}`")
    st.write(f"**Ollama host:** `{OLLAMA_HOST}`")
    stats = get_usage_stats()
    st.metric("Live model calls", stats["requests"])
    st.metric("Response cache hits", stats["cache_hits"])
    st.caption("The counters cover the current Python process.")

    st.divider()
    st.header("Corpus")
    if HOTPOT_QUESTIONS_PATH.exists():
        hotpot_qs = [
            json.loads(line)
            for line in HOTPOT_QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        bridge_n = sum(1 for q in hotpot_qs if q.get("type") == "bridge")
        comp_n = sum(1 for q in hotpot_qs if q.get("type") == "comparison")
        st.metric("HotpotQA questions", len(hotpot_qs))
        st.caption(f"{bridge_n} bridge · {comp_n} comparison")
    else:
        st.warning("HotpotQA subset not built yet.\nRun: `python scripts/build_hotpot_subset.py`")
        hotpot_qs = []

# ── Question selection ────────────────────────────────────────────────────────
cache_data = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}

# Always build the full question list from HotpotQA subset (all 25 questions).
# Fall back to demo_cache keys if the JSONL hasn't been built yet.
if hotpot_qs:
    dropdown_options = [q["question"] for q in hotpot_qs]
    dropdown_labels = {
        q["question"]: f"{q['question']}  [{q['type']} / {q['level']}]"
        for q in hotpot_qs
    }
else:
    dropdown_options = list(cache_data.keys())
    dropdown_labels = {q: q for q in dropdown_options}

default_fallback = (
    dropdown_options[0]
    if dropdown_options
    else "Who discovered penicillin and where was he born?"
)

# Show all questions in the dropdown — cached mode affects how the pipeline runs,
# not which questions are available to pick.
if dropdown_options:
    selected = st.selectbox(
        "Demo question (cached)" if use_cached else "Choose a HotpotQA question",
        options=dropdown_options,
        format_func=lambda q: dropdown_labels.get(q, q),
    )
    question = st.text_input("Question", value=selected, disabled=use_cached)
else:
    question = st.text_input("Question", value=default_fallback)

# Show a badge indicating whether the selected question has a cached run ready
if use_cached:
    if question in cache_data:
        st.success("✅ Cached run available — will replay instantly without Ollama.")
    else:
        st.info("ℹ️ No cached run for this question — will run live via Ollama when you click Run pipeline.")

run = st.button("Run pipeline", type="primary")


# ── Render helpers ────────────────────────────────────────────────────────────
def render_type_legend() -> None:
    dots = " &nbsp;&nbsp; ".join(
        f'<span style="color:{color}; font-size:1.1em;">●</span> {node_type}'
        for node_type, color in TYPE_COLORS.items()
    )
    st.markdown(dots, unsafe_allow_html=True)


def render_graph_section(subproblems: list[dict], n_components: int = 1) -> None:
    st.subheader("2. Logic dependency graph")

    # A2 — warn if the graph has isolated components.
    if n_components > 1:
        st.warning(
            f"⚠️ The dependency graph has **{n_components} disconnected components**. "
            "Some subproblems appear logically isolated from each other — the "
            "decomposition may benefit from refinement."
        )

    graph, _ = build_graph(subproblems)
    try:
        html = render_graph_interactive(graph)
        components.html(html, height=520, scrolling=False)
        st.caption("Hover a node to see the full subproblem text. Drag nodes to rearrange.")
    except Exception:
        st.image(render_graph(graph), use_container_width=True)
    render_type_legend()


def render_doc_with_badge(doc) -> None:
    if isinstance(doc, dict) and "score" in doc:
        title_part = f"**{doc['title']}** — " if doc.get("title") else ""
        st.info(f"{title_part}{doc.get('text', '')}  \n`match: {doc['score']:.2f}`")
    elif isinstance(doc, dict):
        st.info(doc.get("text", ""))
    else:
        st.info(doc)


def render_reward_section(reward: dict, n_nodes: int) -> None:
    """A6 — display composite score + all four named component tiles."""
    st.subheader("5. Reward (4 components)")

    # Composite score + progress bar.
    score = reward.get("score", 0.0)
    normalized = max(0.0, min(1.0, (score + 0.25) / 0.5))  # map [-0.25, 0.25] → [0, 1]
    st.metric("Composite reward score", f"{score:.4f}")
    st.progress(normalized)

    # Individual component tiles.
    comps = reward.get("components", {})
    if comps:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "🔍 Retrieval presence",
            f"{comps.get('retrieval_presence', 0):.2f}",
            help="Fraction of nodes with at least one supporting passage.",
        )
        col2.metric(
            "⚡ Token efficiency",
            f"{comps.get('token_efficiency', 0):.2f}",
            help="1 − (completion_tokens / 2048 token budget).",
        )
        col3.metric(
            "✅ Logical consistency",
            f"{comps.get('logical_consistency', 0):.2f}",
            help="1.0 if no contradictions detected across nodes, 0.0 otherwise.",
        )
        col4.metric(
            "🧠 Grounding",
            f"{comps.get('grounding', 0):.2f}",
            help="Fraction of final-answer sentences grounded in retrieved evidence.",
        )

    st.caption(reward.get("explanation", ""))
    st.warning(
        "This score is a rule-based stand-in for PPO, not a learned reward or "
        "trained policy. See reward.py for the full formula."
    )


# ── Pipeline execution ────────────────────────────────────────────────────────
if run:
    try:
        # Use cached result if toggle is ON AND the question has a cached run.
        # Otherwise always run live (even if toggle is ON — user picked an uncached question).
        run_from_cache = use_cached and question in cache_data

        if run_from_cache:
            result = cache_data[question]
            n_components = result.get("n_components", 1)  # A2

            st.subheader("1. Decomposition")
            for item in result["subproblems"]:
                deps = ", ".join(item.get("depends_on", [])) or "none"
                st.markdown(f"**{item['id']}** — {item['text']}  \nType: `{item.get('type')}` • Depends on: `{deps}`")

            render_graph_section(result["subproblems"], n_components=n_components)

            st.subheader("3. Sequential node resolution")
            for node_id in result["order"]:
                item = next(x for x in result["subproblems"] if x["id"] == node_id)
                contradiction_flag = result.get("node_contradictions", {}).get(node_id, False)
                label = f"{node_id}: {item['text']}"
                if contradiction_flag:
                    label += "  ⚠️ contradiction"
                with st.expander(label, expanded=True):
                    st.markdown("**Retrieved evidence**")
                    for doc in result["retrieved_docs"].get(node_id, []):
                        render_doc_with_badge(doc)
                    st.markdown("**Intermediate answer**")
                    st.write(result["intermediate_answers"].get(node_id, ""))

            st.subheader("4. Final answer")
            st.success(result["final_answer"])
            render_reward_section(result["reward"], len(result["subproblems"]))

        else:
            # Live path — run full pipeline with Ollama
            subproblems: list[dict] = []
            node_placeholders: dict = {}
            status_area = st.container()
            n_components = 1  # A2 default

            with st.spinner("Running the reasoning pipeline…"):
                for event in run_pipeline_stream(question):
                    if event["type"] == "decomposition":
                        subproblems = event["subproblems"]
                        n_components = event.get("n_components", 1)  # A2

                        st.subheader("1. Decomposition")
                        for item in subproblems:
                            deps = ", ".join(item.get("depends_on", [])) or "none"
                            st.markdown(
                                f"**{item['id']}** — {item['text']}  \n"
                                f"Type: `{item.get('type')}` • Depends on: `{deps}`"
                            )
                        render_graph_section(subproblems, n_components=n_components)

                        st.subheader("3. Sequential node resolution")
                        with status_area:
                            for item in subproblems:
                                node_placeholders[item["id"]] = st.empty()
                                node_placeholders[item["id"]].markdown(f"⏳ `{item['id']}` queued")

                    elif event["type"] == "node":
                        node_id = event["node_id"]
                        node_placeholders[node_id].markdown(f"🔄 Resolving `{node_id}`…")
                        contradiction_flag = event.get("contradiction", False)  # A4
                        label = f"{node_id}: {event['subproblem']['text']}"
                        if contradiction_flag:
                            label += "  ⚠️ contradiction detected"
                        with st.expander(label, expanded=False):
                            st.markdown("**Retrieved evidence**")
                            for doc in event["docs"]:
                                render_doc_with_badge(doc)
                            st.markdown("**Intermediate answer**")
                            st.write(event["answer"])
                        node_placeholders[node_id].markdown(f"✅ `{node_id}` resolved")

                    elif event["type"] == "final":
                        st.subheader("4. Final answer")
                        st.success(event["final_answer"])
                        render_reward_section(event["reward"], len(subproblems))

    except Exception as exc:
        st.error(str(exc))
        st.exception(exc)
