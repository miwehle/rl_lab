"""Static plotting for activity-based network layouts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from nn_viz.layout import Edge, NetworkLayout, Node

_NODE_SIZE = 78.0
_LAYER_Y = {"out": 0.0, "h2": 0.25, "h1": 0.5}


def plot_network_layout(layout: NetworkLayout, *, output_path: str | Path | None = None):
    """Plot a compact static network layout and optionally save it."""
    fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
    display_nodes = _display_nodes(layout.nodes)
    nodes = {(node.layer, node.index): node for node in display_nodes}
    _draw_edges(ax, layout.edges, nodes)
    _draw_nodes(ax, display_nodes)
    _label_outputs(ax, display_nodes)
    _label_hidden_nodes(ax, display_nodes)
    ax.set_xlim(_x_limits(display_nodes))
    ax.set_ylim(2.35, -0.35)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _display_nodes(nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    output_nodes = [node for node in nodes if node.layer == "out"]
    output_span = _span(output_nodes)
    h2_nodes = [node for node in nodes if node.layer == "h2"]
    h1_nodes = [node for node in nodes if node.layer == "h1"]
    hidden_frame = _hidden_frame(h2_nodes, fallback_span=output_span)
    output_ordered = sorted(output_nodes, key=lambda node: (node.x, node.index))
    h2_ordered = tuple(node for group in _output_groups(h2_nodes, output_ordered) for node in group)
    h2_display_nodes = _equidistant_nodes(h2_ordered, center=hidden_frame[2], spacing=hidden_frame[3], sort=False)
    return (
        _group_start_nodes(output_ordered, h2_nodes, h2_display_nodes, fallback_span=output_span)
        + h2_display_nodes
        + _equidistant_nodes(h1_nodes, center=hidden_frame[2], spacing=hidden_frame[3])
    )


def _hidden_frame(nodes: list[Node], *, fallback_span: tuple[float, float]) -> tuple[float, float, float, float]:
    if len(nodes) < 2:
        left, right = fallback_span
        center = (left + right) / 2
        return left, right, center, right - left
    center = _center(nodes)
    original_left, original_right = _span(nodes)
    spacing = (original_right - original_left) / (len(nodes) - 1) / 2
    width = spacing * (len(nodes) - 1)
    return center - width / 2, center + width / 2, center, spacing


def _spread_nodes(nodes: list[Node], *, left: float, right: float) -> tuple[Node, ...]:
    if not nodes:
        return ()
    ordered = sorted(nodes, key=lambda node: (node.x, node.index))
    if len(ordered) == 1:
        xs = [(left + right) / 2]
    else:
        xs = np.linspace(left, right, num=len(ordered))
    return tuple(_display_node(node, x=float(x)) for node, x in zip(ordered, xs))


def _group_start_nodes(
    nodes: list[Node],
    original_source_nodes: list[Node],
    display_source_nodes: tuple[Node, ...],
    *,
    fallback_span: tuple[float, float],
) -> tuple[Node, ...]:
    if not nodes:
        return ()
    if len(display_source_nodes) < len(nodes):
        left, right = _span(list(display_source_nodes)) if display_source_nodes else fallback_span
        return _spread_nodes(nodes, left=left, right=right)
    ordered = sorted(nodes, key=lambda node: (node.x, node.index))
    display_by_index = {node.index: node for node in display_source_nodes}
    blocks = _output_groups(original_source_nodes, ordered)
    xs = [display_by_index[block[0].index].x for block in blocks]
    return tuple(_display_node(node, x=float(x)) for node, x in zip(ordered, xs))


def _output_groups(nodes: list[Node], output_nodes: list[Node]) -> list[list[Node]]:
    groups = [[node for node in nodes if node.output_group == output.index] for output in output_nodes]
    if all(groups):
        return groups
    return [list(group) for group in np.array_split(nodes, len(output_nodes))]


def _equidistant_nodes(
    nodes: list[Node] | tuple[Node, ...], *, center: float, spacing: float, sort: bool = True
) -> tuple[Node, ...]:
    if not nodes:
        return ()
    ordered = sorted(nodes, key=lambda node: (node.x, node.index)) if sort else list(nodes)
    if len(ordered) == 1:
        xs = [center]
    else:
        offsets = (np.arange(len(ordered)) - (len(ordered) - 1) / 2) * spacing
        xs = center + offsets
    return tuple(_display_node(node, x=float(x)) for node, x in zip(ordered, xs))


def _display_node(node: Node, *, x: float) -> Node:
    return replace(node, x=x, y=_LAYER_Y[node.layer])


def _span(nodes: list[Node]) -> tuple[float, float]:
    if not nodes:
        return 0.0, 1.0
    xs = [node.x for node in nodes]
    return min(xs), max(xs)


def _center(nodes: list[Node]) -> float:
    left, right = _span(nodes)
    return (left + right) / 2


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
            [source.x, target.x],
            [source.y, target.y],
            color=color,
            linewidth=width,
            alpha=alpha,
            zorder=1,
        )


def _draw_nodes(ax, nodes: tuple[Node, ...]) -> None:
    max_activity = max((node.activity for node in nodes if node.layer != "out"), default=1.0)
    for layer, color in [("h1", "#8a8f98"), ("h2", "#2b6cb0"), ("out", "#dd6b20")]:
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
            ax.text(node.x, node.y + 0.07, str(node.index), ha="center", va="center", fontsize=6, color="#111827")


def _x_limits(nodes: tuple[Node, ...]) -> tuple[float, float]:
    values = [node.x for node in nodes]
    return min(values) - 0.35, max(values) + 0.35
