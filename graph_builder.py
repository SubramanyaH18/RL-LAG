"""Build and render the logic dependency graph.

Upgrades (Track A):
  A2 — build_graph() checks for disconnected components after the DAG check
       and emits a warnings.warn so app.py can surface it to the user.
       Returns (graph, n_components) instead of just the graph.
"""
from __future__ import annotations

import textwrap
import warnings
from io import BytesIO

import networkx as nx
# matplotlib and PIL are imported lazily inside render_graph() so that
# importing graph_builder (e.g. in tests) doesn't hard-require them.

# Color-coded by subproblem type so the DAG communicates structure, not just order.
TYPE_COLORS = {
    "factual": "#4C9AFF",
    "relational": "#57D9A3",
    "comparative": "#FFAB00",
    "temporal": "#C77DFF",
}
DEFAULT_COLOR = "#4C9AFF"


def build_graph(subproblems: list[dict]) -> tuple[nx.DiGraph, int]:
    """Build a directed acyclic graph from a list of subproblem dicts.

    Returns
    -------
    graph : nx.DiGraph
    n_components : int
        Number of weakly connected components. A value > 1 means the
        decomposition produced isolated sub-graphs — a warning is also emitted
        via warnings.warn so callers can surface it to the user (A2).
    """
    graph = nx.DiGraph()
    ids = {item["id"] for item in subproblems}
    for item in subproblems:
        graph.add_node(item["id"], **item)
    for item in subproblems:
        for dependency in item.get("depends_on", []):
            if dependency not in ids:
                raise ValueError(f"Unknown dependency '{dependency}' for node '{item['id']}'")
            graph.add_edge(dependency, item["id"])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The decomposition produced a cycle; retry query decomposition.")

    # A2 — disconnected-component check.
    n_components = nx.number_weakly_connected_components(graph)
    if n_components > 1:
        warnings.warn(
            f"The dependency graph has {n_components} disconnected components. "
            "Some subproblems may be logically isolated — consider refining the "
            "decomposition.",
            stacklevel=2,
        )

    return graph, n_components


def _node_color(graph: nx.DiGraph, node: str) -> str:
    node_type = str(graph.nodes[node].get("type", "")).lower()
    return TYPE_COLORS.get(node_type, DEFAULT_COLOR)


def render_graph(graph: nx.DiGraph):
    """Static matplotlib fallback. Labels are wrapped and nodes sized to fit
    the full text so nothing is clipped at the figure edge.

    Returns a PIL.Image.Image.
    """
    import matplotlib.pyplot as plt  # lazy import — only needed for rendering
    from PIL import Image  # lazy import

    full_labels = {node: graph.nodes[node].get("text", "") for node in graph.nodes}
    wrapped_labels = {
        node: f"{node}\n" + textwrap.fill(text, width=14)
        for node, text in full_labels.items()
    }
    # Size nodes to roughly fit their wrapped label so text doesn't overflow the circle.
    node_sizes = [max(3200, len(full_labels[node]) * 55) for node in graph.nodes]
    node_colors = [_node_color(graph, node) for node in graph.nodes]

    figure = plt.figure(figsize=(14, 8))
    positions = nx.spring_layout(graph, seed=42, k=1.4)
    nx.draw_networkx(
        graph,
        positions,
        labels=wrapped_labels,
        node_size=node_sizes,
        node_color=node_colors,
        font_size=8,
        font_color="black",
        arrows=True,
        arrowsize=18,
        edgecolors="#1b1f27",
        linewidths=1.2,
    )
    plt.axis("off")
    plt.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).copy()


def render_graph_interactive(graph: nx.DiGraph) -> str:
    """Interactive pyvis HTML network. Full subproblem text shows on hover,
    so long labels never get clipped the way the static image did."""
    from pyvis.network import Network

    net = Network(
        height="500px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="white",
    )
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -12000,
          "springLength": 160,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      },
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "color": {"color": "#5c6370"},
        "smooth": false
      }
    }
    """)

    for node in graph.nodes:
        full_text = graph.nodes[node].get("text", "")
        node_type = graph.nodes[node].get("type", "factual")
        truncated = full_text if len(full_text) <= 42 else full_text[:39] + "..."
        net.add_node(
            node,
            label=f"{node}: {truncated}",
            title=full_text,
            color=_node_color(graph, node),
            shape="dot",
            size=22,
            font={"size": 14, "color": "white"},
        )
    for source, target in graph.edges:
        net.add_edge(source, target)

    return net.generate_html(notebook=False)
