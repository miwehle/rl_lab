"""Shared edge construction and state-based visibility."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from nn_viz._rendering import NetworkState, source_value
from nn_viz.layout import Edge, NetworkLayout

EDGE_EFFECT_QUANTILE_DEFAULT = 0.95


def weight_arrays_from_q_net(q_net) -> dict[str, np.ndarray]:
    return {
        "w1": q_net.layer1.weight.detach().cpu().numpy().astype(np.float32, copy=True),
        "w2": q_net.layer2.weight.detach().cpu().numpy().astype(np.float32, copy=True),
        "w3": q_net.layer3.weight.detach().cpu().numpy().astype(np.float32, copy=True),
    }


def network_edges_from_arrays(weights: Mapping[str, np.ndarray]) -> tuple[Edge, ...]:
    return (
        _edges_from_matrix("in", "h1", np.asarray(weights["w1"]))
        + _edges_from_matrix("h1", "h2", np.asarray(weights["w2"]))
        + _edges_from_matrix("h2", "out", np.asarray(weights["w3"]))
    )


def network_edges_from_q_net(q_net) -> tuple[Edge, ...]:
    return network_edges_from_arrays(weight_arrays_from_q_net(q_net))


def network_edges_from_trace(trace: Mapping[str, np.ndarray], layout: NetworkLayout) -> tuple[Edge, ...]:
    if all(key in trace for key in ("w1", "w2", "w3")):
        return network_edges_from_arrays(trace)
    return layout.edges


def select_edges_by_effect(
    edges: tuple[Edge, ...], state: NetworkState, edge_effect_quantile: float
) -> tuple[Edge, ...]:
    if not 0.0 <= edge_effect_quantile <= 1.0:
        raise ValueError("edge_effect_quantile must be in [0, 1]")
    selected = []
    for group in _edge_groups(edges).values():
        effects = np.asarray([_edge_effect(edge, state) for edge in group], dtype=np.float32)
        positive = effects > 0.0
        if not np.any(positive):
            continue
        threshold = float(np.quantile(effects[positive], edge_effect_quantile))
        selected.extend(edge for edge, effect in zip(group, effects) if effect > 0.0 and effect >= threshold)
    return tuple(selected)


def scales_with_edge_weight_scale(scales: Mapping[str, Any] | None, edges: tuple[Edge, ...]) -> dict[str, Any]:
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


def _edge_effect(edge: Edge, state: NetworkState) -> float:
    return abs(source_value(edge, state) * edge.weight)


def _edge_groups(edges: tuple[Edge, ...]) -> dict[tuple[str, str], list[Edge]]:
    groups: dict[tuple[str, str], list[Edge]] = {}
    for edge in edges:
        groups.setdefault((edge.source_layer, edge.target_layer), []).append(edge)
    return groups
