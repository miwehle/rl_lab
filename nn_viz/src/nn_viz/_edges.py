"""Shared edge construction and state-based visibility."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from nn_viz._rendering import NetworkState, source_value
from nn_viz.layout import Edge, NetworkLayout

EDGE_CONTRIBUTORS_PER_TARGET_DEFAULT = 6


def weight_arrays_from_q_net(q_net) -> dict[str, np.ndarray]:
    return {
        "w1": q_net.layer1.weight.detach().cpu().numpy().astype(np.float32, copy=True),
        "w2": q_net.layer2.weight.detach().cpu().numpy().astype(np.float32, copy=True),
        "w3": q_net.layer3.weight.detach().cpu().numpy().astype(np.float32, copy=True),
    }


def _network_edges_from_arrays(weights: Mapping[str, np.ndarray]) -> tuple[Edge, ...]:
    return (
        _edges_from_matrix("in", "h1", np.asarray(weights["w1"]))
        + _edges_from_matrix("h1", "h2", np.asarray(weights["w2"]))
        + _edges_from_matrix("h2", "out", np.asarray(weights["w3"]))
    )


def network_edges_from_q_net(q_net) -> tuple[Edge, ...]:
    return _network_edges_from_arrays(weight_arrays_from_q_net(q_net))


def network_edges_from_trace(trace: Mapping[str, np.ndarray], layout: NetworkLayout) -> tuple[Edge, ...]:
    if all(key in trace for key in ("w1", "w2", "w3")):
        return _network_edges_from_arrays(trace)
    return layout.edges


def select_edges_by_target_contributors(
    edges: tuple[Edge, ...], state: NetworkState, edge_contributors_per_target: int
) -> tuple[Edge, ...]:
    """Return the frame-specific edge subset with the strongest contributors per target.

    This is the core function of the module. It starts from candidate weight edges
    and selects, for each target neuron, the currently strongest incoming positive
    and negative contributors.

    Args:
        edges: Candidate edges to choose from. Usually this is the full network
            edge set reconstructed from the current q_net weights or from a saved
            trace.
        state: Current network state. Source activations are read from this state
            and combined with edge weights.
        edge_contributors_per_target: Maximum number of incoming edges selected
            for each target neuron. The budget is split between positive and
            negative contributors according to their total contribution strength.
            A value of 0 selects no edges.

    Returns:
        The selected edges for the current state. The returned tuple contains only
        edges whose current contribution is non-zero and among the strongest
        contributors for their target neuron.
    """
    if edge_contributors_per_target < 0:
        raise ValueError("edge_contributors_per_target must be >= 0")
    if edge_contributors_per_target == 0:
        return ()
    selected = []
    for target_edges in _edge_groups_by_target(edges).values():
        contributions = [(edge, source_value(edge, state) * edge.weight) for edge in target_edges]
        selected.extend(_select_target_contributors(contributions, edge_contributors_per_target))
    return tuple(selected)


def scales_with_edge_weight_scale(
    scales: Mapping[str, Any] | None, edges: tuple[Edge, ...]
) -> dict[str, Any]:
    render_scales = dict(scales or {})
    weights = np.asarray([abs(edge.weight) for edge in edges], dtype=np.float32)
    render_scales["weight"] = float(np.percentile(weights, 95)) if weights.size else 1.0
    return render_scales


def _edges_from_matrix(source_layer: str, target_layer: str, weights: np.ndarray) -> tuple[Edge, ...]:
    edges = []
    for target in range(weights.shape[0]):
        for source in range(weights.shape[1]):
            weight = float(weights[target, source])
            if weight != 0.0:
                relevance = abs(weight)
                edges.append(Edge(source_layer, source, target_layer, target, weight, relevance, relevance))
    return tuple(edges)


def _select_target_contributors(
    contributions: list[tuple[Edge, float]], edge_contributors_per_target: int
) -> list[Edge]:
    positive = _sorted_contribution_group(contributions, positive=True)
    negative = _sorted_contribution_group(contributions, positive=False)
    pos_sum = sum(effect for _, effect in positive)
    neg_sum = sum(effect for _, effect in negative)
    total = pos_sum + neg_sum
    if total == 0.0:
        return []

    k_pos = int(round(edge_contributors_per_target * pos_sum / total))
    k_neg = edge_contributors_per_target - k_pos
    return [edge for edge, _ in positive[:k_pos]] + [edge for edge, _ in negative[:k_neg]]


def _sorted_contribution_group(
    contributions: list[tuple[Edge, float]], *, positive: bool
) -> list[tuple[Edge, float]]:
    group = [
        (edge, abs(contribution))
        for edge, contribution in contributions
        if (contribution > 0.0 if positive else contribution < 0.0)
    ]
    return sorted(group, key=lambda item: (-item[1], item[0].source_layer, item[0].source_index))


def _edge_groups_by_target(edges: tuple[Edge, ...]) -> dict[tuple[str, int], list[Edge]]:
    groups: dict[tuple[str, int], list[Edge]] = {}
    for edge in edges:
        groups.setdefault((edge.target_layer, edge.target_index), []).append(edge)
    return groups
