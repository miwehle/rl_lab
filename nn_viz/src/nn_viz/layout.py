"""Compute stable, activity-based layouts for Elise-like DQN networks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dqn.model import DQN
from nn_viz.activations import ACTION_LABELS, ACTION_ORDER, ActivationRollouts

INPUT_LABELS = ("x", "y", "vx", "vy", "ang", "vang", "ftl", "ftr", "ax", "ay")


@dataclass(frozen=True)
class Node:
    """
    activity: rollout mean
    output_group: the output action used to group H2 nodes.
    """

    layer: str
    index: int
    label: str
    x: float
    y: float
    activity: float
    output_group: int | None = None


@dataclass(frozen=True)
class Edge:
    """
    relevance: rollout mean of abs(source activation * weight)
    specificity: relevance - mean(other relevances)
    """

    source_layer: str
    source_index: int
    target_layer: str
    target_index: int
    weight: float
    relevance: float
    specificity: float


@dataclass(frozen=True)
class NetworkLayout:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


def compute_activity_layout(
    rollouts: ActivationRollouts,
    q_net: DQN,
    *,
    top_edges_per_target: int = 3,
    output_edges_per_target: int = 5,
) -> NetworkLayout:
    """Place hidden neurons by their mean activated contribution over the rollout frames."""
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
    h2_nodes = _hidden2_nodes(rollouts.h2, h2_output_specificity, output_nodes)
    h1_nodes = _hidden1_nodes(rollouts.h1, h1_to_h2, h2_nodes)
    input_nodes = _input_nodes(rollouts.observations)
    edges = (
        _top_edges("h2", "out", h2_to_out, h2_output_specificity, w3, output_edges_per_target)
        + _top_edges("h1", "h2", h1_to_h2, h1_to_h2, w2, top_edges_per_target)
        + _top_edges("in", "h1", input_to_h1, input_to_h1, w1, top_edges_per_target)
    )
    return NetworkLayout(nodes=output_nodes + h2_nodes + h1_nodes + input_nodes, edges=tuple(edges))


def _mean_abs_contribution(
    source_activations: np.ndarray, target_by_source_weights: np.ndarray
) -> np.ndarray:
    return np.mean(
        np.abs(source_activations[:, np.newaxis, :] * target_by_source_weights[np.newaxis, :, :]), axis=0
    )


def _target_specificity(relevance: np.ndarray) -> np.ndarray:
    if relevance.shape[0] == 1:
        return relevance.copy()
    other_sum = np.sum(relevance, axis=0, keepdims=True) - relevance
    return relevance - other_sum / (relevance.shape[0] - 1)


def _output_nodes(q_values: np.ndarray) -> tuple[Node, ...]:
    output_x = {action: float(position) for position, action in enumerate(ACTION_ORDER)}
    return tuple(
        Node("out", action, ACTION_LABELS[action], output_x[action], 0.0, float(np.mean(q_values[:, action])))
        for action in ACTION_ORDER
    )


def _input_nodes(observations: np.ndarray) -> tuple[Node, ...]:
    return tuple(
        Node("in", index, label, float(index), 1.0, float(np.mean(np.abs(observations[:, index]))))
        for index, label in enumerate(INPUT_LABELS)
    )


def _hidden2_nodes(h2: np.ndarray, h2_to_out: np.ndarray, output_nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    output_x = {node.index: node.x for node in output_nodes}
    groups: dict[int, list[int]] = {node.index: [] for node in output_nodes}
    for index in range(h2.shape[1]):
        groups[int(np.argmax(h2_to_out[:, index]))].append(index)

    nodes: list[Node] = []
    for output, indexes in groups.items():
        ordered = sorted(indexes, key=lambda index: h2_to_out[output, index], reverse=True)
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


def _hidden1_nodes(h1: np.ndarray, h1_to_h2: np.ndarray, h2_nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    h2_x = np.asarray([node.x for node in sorted(h2_nodes, key=lambda node: node.index)], dtype=np.float64)
    center = float(np.mean(h2_x))
    nodes = []
    for index in range(h1.shape[1]):
        strengths = h1_to_h2[:, index]
        total = float(np.sum(strengths))
        x = center if total == 0.0 else float(np.sum(strengths * h2_x) / total)
        nodes.append(Node("h1", index, f"H1-{index}", x, 2.0, float(np.mean(h1[:, index]))))
    return tuple(nodes)


def _top_edges(
    source_layer: str,
    target_layer: str,
    relevance: np.ndarray,
    specificity: np.ndarray,
    weights: np.ndarray,
    top_edges_per_target: int,
) -> tuple[Edge, ...]:
    edges = []
    for target in range(relevance.shape[0]):
        source_indexes = np.argsort(specificity[target])[-top_edges_per_target:][::-1]
        for source in source_indexes:
            if specificity[target, source] > 0.0:
                edges.append(
                    Edge(
                        source_layer,
                        int(source),
                        target_layer,
                        int(target),
                        float(weights[target, source]),
                        float(relevance[target, source]),
                        float(specificity[target, source]),
                    )
                )
    return tuple(edges)


def _centered_offsets(count: int, *, spacing: float) -> list[float]:
    return [(index - (count - 1) / 2) * spacing for index in range(count)]
