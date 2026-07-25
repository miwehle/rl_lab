"""Compute semantic-anchor 3D layouts for Elise-like DQN networks."""

from __future__ import annotations

import numpy as np

from dqn.model import DQN
from nn_viz.activations import ACTION_LABELS, ACTION_ORDER, ActivationRollouts
from nn_viz.layout.types import INPUT_LABELS, Edge, NetworkLayout, Node

_INPUT_Z = 0.0
_H1_Z = 1.0
_H2_Z = 2.0
_OUTPUT_Z = 3.0
_MIN_NODE_DISTANCE_DEFAULT = 0.14
_COLLISION_ITERATIONS = 20

_INPUT_ANCHORS = {
    6: (-1.5, 1.5),  # ftl
    7: (1.5, 1.5),  # ftr
    1: (-1.5, 0.5),  # y
    3: (0.0, 0.5),  # vy
    9: (1.5, 0.5),  # ay
    0: (-1.5, -0.5),  # x
    2: (0.0, -0.5),  # vx
    8: (1.5, -0.5),  # ax
    4: (-1.5, -1.5),  # ang
    5: (0.0, -1.5),  # vang
}

_OUTPUT_ANCHORS = {
    1: (-1.5, 1.5),  # left
    3: (1.5, 1.5),  # right
    2: (-1.5, 0.5),  # up
    0: (1.5, 0.5),  # noop
}


def compute_layout(
    rollouts: ActivationRollouts,
    q_net: DQN,
    *,
    top_edges_per_target: int = 3,
    output_edges_per_target: int = 10,
    min_node_distance: float = _MIN_NODE_DISTANCE_DEFAULT,
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

    w1 = q_net.layer1.weight.detach().cpu().numpy()
    w2 = q_net.layer2.weight.detach().cpu().numpy()
    w3 = q_net.layer3.weight.detach().cpu().numpy()
    input_anchors = _centered_input_anchors()
    output_anchors = _shift_output_anchors(_input_center())

    input_layer_nodes = _input_nodes(rollouts.observations, input_anchors)
    h1_nodes = _hidden_nodes(
        "h1",
        rollouts.h1,
        source_positions=np.asarray([input_anchors[index] for index in range(len(INPUT_LABELS))]),
        weights=w1,
        z=_H1_Z,
        min_node_distance=min_node_distance,
    )
    h1_positions = np.asarray([(node.x, node.y) for node in sorted(h1_nodes, key=lambda node: node.index)])
    h2_nodes = _hidden_nodes(
        "h2",
        rollouts.h2,
        source_positions=h1_positions,
        weights=w2,
        z=_H2_Z,
        min_node_distance=min_node_distance,
    )
    output_layer_nodes = _output_nodes(rollouts.q_values, output_anchors)
    return NetworkLayout(
        nodes=input_layer_nodes + h1_nodes + h2_nodes + output_layer_nodes,
        edges=(
            _top_weight_edges("in", "h1", w1, top_edges_per_target)
            + _top_weight_edges("h1", "h2", w2, top_edges_per_target)
            + _top_weight_edges("h2", "out", w3, output_edges_per_target)
        ),
    )


def _input_nodes(observations: np.ndarray, input_anchors: dict[int, tuple[float, float]]) -> tuple[Node, ...]:
    return tuple(
        Node(
            "in",
            index,
            label,
            *input_anchors[index],
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
    weights: np.ndarray,
    z: float,
    min_node_distance: float,
) -> tuple[Node, ...]:
    fallback = np.mean(source_positions, axis=0)
    positions = np.asarray(
        [
            _weighted_mean(source_positions, np.abs(weights[index]), fallback)
            for index in range(weights.shape[0])
        ]
    )
    positions = _separate_points(positions, min_node_distance)
    nodes = []
    for index in range(weights.shape[0]):
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


def _output_nodes(q_values: np.ndarray, output_anchors: dict[int, tuple[float, float]]) -> tuple[Node, ...]:
    return tuple(
        Node(
            "out",
            action,
            ACTION_LABELS[action],
            *output_anchors[action],
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


def _separate_points(positions: np.ndarray, min_distance: float) -> np.ndarray:
    if len(positions) < 2 or min_distance == 0.0:
        return positions
    adjusted = positions.astype(np.float64, copy=True)
    center = np.mean(adjusted, axis=0)
    for _ in range(_COLLISION_ITERATIONS):
        moved = False
        for left in range(len(adjusted) - 1):
            for right in range(left + 1, len(adjusted)):
                delta = adjusted[left] - adjusted[right]
                distance = float(np.linalg.norm(delta))
                if distance >= min_distance:
                    continue
                direction = _separation_direction(delta, distance, left, right)
                push = 0.5 * (min_distance - distance)
                adjusted[left] += direction * push
                adjusted[right] -= direction * push
                moved = True
        adjusted += center - np.mean(adjusted, axis=0)
        if not moved:
            break
    return adjusted


def _separation_direction(delta: np.ndarray, distance: float, left: int, right: int) -> np.ndarray:
    if distance > 1e-12:
        return delta / distance
    angle = ((left + 1) * 12.9898 + (right + 1) * 78.233) % (2.0 * np.pi)
    return np.asarray([np.cos(angle), np.sin(angle)])


def _centered_input_anchors() -> dict[int, tuple[float, float]]:
    center = _input_center()
    return {index: _shift(position, center) for index, position in _INPUT_ANCHORS.items()}


def _shift_output_anchors(center: np.ndarray) -> dict[int, tuple[float, float]]:
    return {action: _shift(position, center) for action, position in _OUTPUT_ANCHORS.items()}


def _input_center() -> np.ndarray:
    return np.mean(np.asarray([_INPUT_ANCHORS[index] for index in range(len(INPUT_LABELS))]), axis=0)


def _shift(position: tuple[float, float], center: np.ndarray) -> tuple[float, float]:
    shifted = np.asarray(position) - center
    return float(shifted[0]), float(shifted[1])


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
