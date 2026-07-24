"""Data structures for NN layout rendering."""

from __future__ import annotations

from dataclasses import dataclass

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
