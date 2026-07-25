"""Static plotting for activity-based network layouts.

Technical base: matplotlib

Used for manual/static inspection, not for video or trace rendering.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from nn_viz._rendering import display_nodes
from nn_viz.layout import Edge, NetworkLayout, Node

_NODE_SIZE = 78.0


def plot_network_layout(layout: NetworkLayout, *, output_path: str | Path | None = None):
    """Plot a compact static network layout and optionally save it.

    Function: NetworkLayout -> Figure

    Returns:
        The network as Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
    display_layout_nodes = display_nodes(layout.nodes)
    nodes = {(node.layer, node.index): node for node in display_layout_nodes}
    _draw_edges(ax, layout.edges, nodes)
    _draw_nodes(ax, display_layout_nodes)
    _label_outputs(ax, display_layout_nodes)
    _label_hidden_nodes(ax, display_layout_nodes)
    ax.set_xlim(_x_limits(display_layout_nodes))
    ax.set_ylim(2.35, -0.35)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _draw_edges(ax, edges: tuple[Edge, ...], nodes: dict[tuple[str, int], Node]) -> None:
    max_width_value = max((abs(edge.weight) for edge in edges), default=1.0)
    max_alpha_value = max((edge.specificity for edge in edges), default=1.0)
    for edge in edges:
        source = nodes[(edge.source_layer, edge.source_index)]
        target = nodes[(edge.target_layer, edge.target_index)]
        color = "#2f855a" if edge.weight >= 0 else "#b83232"
        width = 0.35 + 2.3 * abs(edge.weight) / max_width_value
        alpha = 0.08 + 0.55 * edge.specificity / max_alpha_value
        ax.plot(
            [source.x, target.x], [source.y, target.y], color=color, linewidth=width, alpha=alpha, zorder=1
        )


def _draw_nodes(ax, nodes: tuple[Node, ...]) -> None:
    max_activity = max((node.activity for node in nodes if node.layer != "out"), default=1.0)
    for layer, color in [("in", "#6b7280"), ("h1", "#8a8f98"), ("h2", "#2b6cb0"), ("out", "#dd6b20")]:
        selected = [node for node in nodes if node.layer == layer]
        if not selected:
            continue
        ax.scatter(
            [node.x for node in selected],
            [node.y for node in selected],
            s=_NODE_SIZE,
            color=[_node_color(node, color, max_activity) for node in selected],
            edgecolors="#111827",
            linewidths=0.35,
            zorder=2,
        )


def _node_color(node: Node, color: str, max_activity: float) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(color))
    pale = np.asarray((0.93, 0.95, 0.97))
    if node.layer == "out":
        return tuple(base)
    if max_activity <= 0.0:
        brightness = 0.0
    else:
        brightness = float(np.sqrt(max(node.activity, 0.0) / max_activity))
    return tuple(pale * (1.0 - brightness) + base * brightness)


def _label_outputs(ax, nodes: tuple[Node, ...]) -> None:
    for node in nodes:
        if node.layer == "out":
            ax.text(node.x, node.y - 0.09, node.label, ha="center", va="center", fontsize=10, weight="bold")


def _label_hidden_nodes(ax, nodes: tuple[Node, ...]) -> None:
    for node in nodes:
        if node.layer in {"h1", "h2"}:
            ax.text(
                node.x, node.y + 0.07, str(node.index), ha="center", va="center", fontsize=6, color="#111827"
            )
        if node.layer == "in":
            ax.text(node.x, node.y + 0.08, node.label, ha="center", va="center", fontsize=7, color="#111827")


def _x_limits(nodes: tuple[Node, ...]) -> tuple[float, float]:
    values = [node.x for node in nodes]
    return min(values) - 0.35, max(values) + 0.35
