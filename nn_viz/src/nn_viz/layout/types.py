"""Data structures for NN layout rendering."""

from __future__ import annotations

from dataclasses import dataclass

INPUT_LABELS = ("x", "y", "vx", "vy", "ang", "vang", "ftl", "ftr", "ax", "ay")


@dataclass(frozen=True)
class Node:
    """
    output_group: the output action used to group H2 nodes.
    activity: rollout mean
    z: 3D layer height; 2D renderers may ignore it.
    """
    layer: str
    index: int
    label: str
    x: float
    y: float
    activity: float
    output_group: int | None = None
    z: float = 0.0


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
    """Network that can be rendered.
    
    nn_viz.video can render it:
    - as PNG and 
    - into a video
    """
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
