"""Compute semantic-anchor 3D layouts for Elise-like DQN networks."""

from __future__ import annotations

import numpy as np

from dqn import DQN
from nn_viz.activations import ACTION_LABELS, ACTION_ORDER, ActivationRollouts
from nn_viz.layout.activity import mean_abs_contribution
from nn_viz.layout.types import INPUT_LABELS, Edge, NetworkLayout, Node

_INPUT_Z = 0.0
_H1_Z = 1.0
_H2_Z = 2.0
_OUTPUT_Z = 3.0
_MIN_NODE_DISTANCE_DEFAULT = 0.14
_EDGE_WEIGHT_QUANTILE_DEFAULT = 0.70
_COLLISION_ITERATIONS = 100
_STIFFNESS_EPS = 1e-6

_INPUT_ANCHORS = {
    6: (-1.0, 2.0),  # ftl
    7: (1.0, 2.0),  # ftr
    1: (-2.0, 1.0),  # y
    3: (0.0, 1.0),  # vy
    9: (2.0, 1.0),  # ay
    0: (-2.0, -1.0),  # x
    2: (0.0, -1.0),  # vx
    8: (2.0, -1.0),  # ax
    4: (-1.0, -2.0),  # ang
    5: (1.0, -2.0),  # vang
}

_OUTPUT_ANCHORS = {
    2: (-1.0, 1.0),  # up
    0: (1.0, 1.0),  # noop
    1: (-1.0, -1.0),  # left
    3: (1.0, -1.0),  # right
}


def compute_layout(
    rollouts: ActivationRollouts,
    q_net: DQN,
    *,
    top_edges_per_target: int = 3,
    output_edges_per_target: int = 10,
    min_node_distance: float = _MIN_NODE_DISTANCE_DEFAULT,
    edge_weight_quantile: float = _EDGE_WEIGHT_QUANTILE_DEFAULT,
) -> NetworkLayout:
    """Place input/output on fixed anchors and hidden nodes by weighted means."""
    if rollouts.frame_count < 1:
        raise ValueError("rollouts must contain at least one frame")
    if top_edges_per_target < 1:
        raise ValueError("top_edges_per_target must be >= 1")
    if output_edges_per_target < 1:
        raise ValueError("output_edges_per_target must be >= 1")
    if min_node_distance < 0.0:
        raise ValueError("min_node_distance must be >= 0")
    if not 0.0 <= edge_weight_quantile <= 1.0:
        raise ValueError("edge_weight_quantile must be in [0, 1]")

    w1 = q_net.layer1.weight.detach().cpu().numpy()
    w2 = q_net.layer2.weight.detach().cpu().numpy()
    w3 = q_net.layer3.weight.detach().cpu().numpy()
    input_to_h1 = mean_abs_contribution(rollouts.observations, w1)
    h1_to_h2 = mean_abs_contribution(rollouts.h1, w2)

    input_layer_nodes = _input_nodes(rollouts.observations)
    h1_nodes = _hidden_nodes(
        "h1",
        rollouts.h1,
        source_positions=np.asarray([_INPUT_ANCHORS[index] for index in range(len(INPUT_LABELS))]),
        placement_weights=input_to_h1,
        z=_H1_Z,
        min_node_distance=min_node_distance,
    )
    h1_positions = np.asarray([(node.x, node.y) for node in sorted(h1_nodes, key=lambda node: node.index)])
    h2_nodes = _hidden_nodes(
        "h2",
        rollouts.h2,
        source_positions=h1_positions,
        placement_weights=h1_to_h2,
        z=_H2_Z,
        min_node_distance=min_node_distance,
    )
    output_layer_nodes = _output_nodes(rollouts.q_values)
    edge_candidates = (
        _top_weight_edges("in", "h1", w1, top_edges_per_target)
        + _top_weight_edges("h1", "h2", w2, top_edges_per_target)
        + _top_weight_edges("h2", "out", w3, output_edges_per_target)
    )
    return NetworkLayout(
        nodes=input_layer_nodes + h1_nodes + h2_nodes + output_layer_nodes,
        edges=_filter_edges_by_weight_quantile(edge_candidates, edge_weight_quantile),
    )


def _input_nodes(observations: np.ndarray) -> tuple[Node, ...]:
    return tuple(
        Node(
            "in",
            index,
            label,
            *_INPUT_ANCHORS[index],
            float(np.mean(np.abs(observations[:, index]))),
            z=_INPUT_Z,
        )
        for index, label in enumerate(INPUT_LABELS)
    )


def _hidden_nodes(
    layer: str,
    activations: np.ndarray,
    *,
    source_positions: np.ndarray,
    placement_weights: np.ndarray,
    z: float,
    min_node_distance: float,
) -> tuple[Node, ...]:
    fallback = np.mean(source_positions, axis=0)
    positions = np.asarray(
        [
            _weighted_mean(source_positions, placement_weights[index], fallback)
            for index in range(placement_weights.shape[0])
        ]
    )
    positions = _separate_points(positions, min_node_distance, _stiffness(placement_weights))
    nodes = []
    for index in range(placement_weights.shape[0]):
        position = positions[index]
        nodes.append(
            Node(
                layer,
                index,
                f"{layer.upper()}-{index}",
                float(position[0]),
                float(position[1]),
                float(np.mean(activations[:, index])),
                z=z,
            )
        )
    return tuple(nodes)


def _output_nodes(q_values: np.ndarray) -> tuple[Node, ...]:
    return tuple(
        Node(
            "out",
            action,
            ACTION_LABELS[action],
            *_OUTPUT_ANCHORS[action],
            float(np.mean(q_values[:, action])),
            z=_OUTPUT_Z,
        )
        for action in ACTION_ORDER
    )


def _weighted_mean(positions: np.ndarray, weights: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total == 0.0:
        return fallback
    return np.sum(positions * weights[:, np.newaxis], axis=0) / total


def _separate_points(positions: np.ndarray, min_distance: float, stiffness: np.ndarray) -> np.ndarray:
    if len(positions) < 2 or min_distance == 0.0:
        return positions
    adjusted = positions.astype(np.float64, copy=True)
    center = _weighted_center(adjusted, stiffness)
    mobility = 1.0 / stiffness
    for _ in range(_COLLISION_ITERATIONS):
        moved = False
        for left in range(len(adjusted) - 1):
            for right in range(left + 1, len(adjusted)):
                delta = adjusted[left] - adjusted[right]
                distance = float(np.linalg.norm(delta))
                if distance >= min_distance:
                    continue
                direction = _separation_direction(delta, distance, left, right)
                overlap = min_distance - distance
                total_mobility = mobility[left] + mobility[right]
                left_share = mobility[left] / total_mobility
                right_share = mobility[right] / total_mobility
                adjusted[left] += direction * overlap * left_share
                adjusted[right] -= direction * overlap * right_share
                moved = True
        adjusted += center - _weighted_center(adjusted, stiffness)
        if not moved:
            break
    return adjusted


def _stiffness(weights: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(weights), axis=1) + _STIFFNESS_EPS


def _weighted_center(positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(positions * weights[:, np.newaxis], axis=0) / float(np.sum(weights))


def _separation_direction(delta: np.ndarray, distance: float, left: int, right: int) -> np.ndarray:
    if distance > 1e-12:
        return delta / distance
    angle = ((left + 1) * 12.9898 + (right + 1) * 78.233) % (2.0 * np.pi)
    return np.asarray([np.cos(angle), np.sin(angle)])


def _top_weight_edges(
    source_layer: str, target_layer: str, weights: np.ndarray, top_edges_per_target: int
) -> tuple[Edge, ...]:
    edges = []
    for target in range(weights.shape[0]):
        source_indexes = np.argsort(np.abs(weights[target]))[-top_edges_per_target:][::-1]
        for source in source_indexes:
            weight = float(weights[target, source])
            if weight == 0.0:
                continue
            relevance = abs(weight)
            edges.append(Edge(source_layer, source, target_layer, target, weight, relevance, relevance))
    return tuple(edges)


def _filter_edges_by_weight_quantile(edges: tuple[Edge, ...], quantile: float) -> tuple[Edge, ...]:
    if not edges or quantile == 0.0:
        return edges
    selected = []
    for group in _edge_groups(edges).values():
        threshold = float(np.quantile([abs(edge.weight) for edge in group], quantile))
        selected.extend(edge for edge in group if abs(edge.weight) >= threshold)
    return tuple(selected)


def _edge_groups(edges: tuple[Edge, ...]) -> dict[tuple[str, str], list[Edge]]:
    groups: dict[tuple[str, str], list[Edge]] = {}
    for edge in edges:
        groups.setdefault((edge.source_layer, edge.target_layer), []).append(edge)
    return groups
