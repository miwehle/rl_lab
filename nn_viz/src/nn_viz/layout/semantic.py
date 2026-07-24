"""Compute semantic-anchor layouts for Elise-like DQN networks."""

from __future__ import annotations

import numpy as np

from dqn.model import DQN
from nn_viz.activations import ACTION_ORDER, ActivationRollouts
from nn_viz.layout.activity import (
    _centered_offsets,
    _input_nodes,
    _mean_abs_contribution,
    _output_nodes,
    _target_specificity,
    _top_edges,
)
from nn_viz.layout.types import NetworkLayout, Node

_INPUT_GROUPS = (
    (0, 2, 8),  # horizontal: x, vx, ax
    (1, 3, 9),  # vertical: y, vy, ay
    (4, 5),  # attitude: angle, angular velocity
    (6, 7),  # contact: left foot, right foot
)


def compute_semantic_layout(
    rollouts: ActivationRollouts,
    q_net: DQN,
    *,
    top_edges_per_target: int = 3,
    output_edges_per_target: int = 5,
) -> NetworkLayout:
    """Sort H1 by input groups and H2 by output actions."""
    if rollouts.frame_count < 1:
        raise ValueError("rollouts must contain at least one frame")
    if top_edges_per_target < 1:
        raise ValueError("top_edges_per_target must be >= 1")
    if output_edges_per_target < 1:
        raise ValueError("output_edges_per_target must be >= 1")

    w1 = q_net.layer1.weight.detach().cpu().numpy()
    w2 = q_net.layer2.weight.detach().cpu().numpy()
    w3 = q_net.layer3.weight.detach().cpu().numpy()
    input_to_h1 = _mean_abs_contribution(rollouts.observations, w1)
    h1_to_h2 = _mean_abs_contribution(rollouts.h1, w2)
    h2_to_out = _mean_abs_contribution(rollouts.h2, w3)
    h2_output_specificity = _target_specificity(h2_to_out)

    output_nodes = _output_nodes(rollouts.q_values)
    h2_nodes = _semantic_hidden2_nodes(rollouts.h2, h2_to_out, output_nodes)
    h1_nodes = _semantic_hidden1_nodes(rollouts.h1, input_to_h1)
    input_nodes = _input_nodes(rollouts.observations)
    edges = (
        _top_edges("h2", "out", h2_to_out, h2_output_specificity, w3, output_edges_per_target)
        + _top_edges("h1", "h2", h1_to_h2, h1_to_h2, w2, top_edges_per_target)
        + _top_edges("in", "h1", input_to_h1, input_to_h1, w1, top_edges_per_target)
    )
    return NetworkLayout(nodes=output_nodes + h2_nodes + h1_nodes + input_nodes, edges=tuple(edges))


def _semantic_hidden1_nodes(h1: np.ndarray, input_to_h1: np.ndarray) -> tuple[Node, ...]:
    activation_frequency = np.mean(h1 > 0.0, axis=0)
    mean_activation = np.mean(h1, axis=0)
    ordered = sorted(
        range(h1.shape[1]),
        key=lambda index: (
            _dominant_input_group(input_to_h1[index]),
            -float(activation_frequency[index]),
            -float(mean_activation[index]),
            index,
        ),
    )
    return tuple(
        Node("h1", index, f"H1-{index}", float(position), 2.0, float(mean_activation[index]))
        for position, index in enumerate(ordered)
    )


def _semantic_hidden2_nodes(h2: np.ndarray, h2_to_out: np.ndarray, output_nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    output_x = {node.index: node.x for node in output_nodes}
    groups: dict[int, list[int]] = {node.index: [] for node in output_nodes}
    for index in range(h2.shape[1]):
        groups[int(np.argmax(h2_to_out[:, index]))].append(index)

    nodes: list[Node] = []
    for output in ACTION_ORDER:
        indexes = groups[output]
        ordered = sorted(indexes, key=lambda index: (-float(h2_to_out[output, index]), index))
        offsets = _centered_offsets(len(ordered), spacing=0.12)
        nodes.extend(
            Node(
                "h2",
                index,
                f"H2-{index}",
                output_x[output] + offsets[position],
                1.0,
                float(np.mean(h2[:, index])),
                output,
            )
            for position, index in enumerate(ordered)
        )
    return tuple(nodes)


def _dominant_input_group(input_relevance: np.ndarray) -> int:
    group_scores = [float(np.sum(input_relevance[list(group)])) for group in _INPUT_GROUPS]
    return int(np.argmax(group_scores))
