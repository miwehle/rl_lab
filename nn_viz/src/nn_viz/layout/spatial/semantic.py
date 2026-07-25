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
) -> NetworkLayout:
    """Place input/output on fixed anchors and hidden nodes by weighted means."""
    if rollouts.frame_count < 1:
        raise ValueError("rollouts must contain at least one frame")
    if top_edges_per_target < 1:
        raise ValueError("top_edges_per_target must be >= 1")
    if output_edges_per_target < 1:
        raise ValueError("output_edges_per_target must be >= 1")

    w1 = q_net.layer1.weight.detach().cpu().numpy()
    w2 = q_net.layer2.weight.detach().cpu().numpy()
    w3 = q_net.layer3.weight.detach().cpu().numpy()

    input_layer_nodes = _input_nodes(rollouts.observations)
    h1_nodes = _hidden_nodes(
        "h1",
        rollouts.h1,
        source_positions=np.asarray([_INPUT_ANCHORS[index] for index in range(len(INPUT_LABELS))]),
        weights=w1,
        z=_H1_Z,
    )
    h1_positions = np.asarray([(node.x, node.y) for node in sorted(h1_nodes, key=lambda node: node.index)])
    h2_nodes = _hidden_nodes("h2", rollouts.h2, source_positions=h1_positions, weights=w2, z=_H2_Z)
    output_layer_nodes = _output_nodes(rollouts.q_values)
    return NetworkLayout(
        nodes=input_layer_nodes + h1_nodes + h2_nodes + output_layer_nodes,
        edges=(
            _top_weight_edges("in", "h1", w1, top_edges_per_target)
            + _top_weight_edges("h1", "h2", w2, top_edges_per_target)
            + _top_weight_edges("h2", "out", w3, output_edges_per_target)
        ),
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
    layer: str, activations: np.ndarray, *, source_positions: np.ndarray, weights: np.ndarray, z: float
) -> tuple[Node, ...]:
    fallback = np.mean(source_positions, axis=0)
    nodes = []
    for index in range(weights.shape[0]):
        position = _weighted_mean(source_positions, np.abs(weights[index]), fallback)
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
