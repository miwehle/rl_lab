import numpy as np

from nn_viz._edges import select_edges_by_effect
from nn_viz._rendering import NetworkState
from nn_viz.layout import Edge


def test_select_edges_by_effect_filters_layerwise():
    state = NetworkState(
        inputs=np.array([10.0, 1.0], dtype=np.float32),
        h1=np.array([2.0, 1.0], dtype=np.float32),
        h2=np.array([3.0, 1.0], dtype=np.float32),
        q_values=np.array([0.0], dtype=np.float32),
        action=0,
    )
    edges = (
        Edge("in", 0, "h1", 0, 1.0, 0.0, 0.0),
        Edge("in", 1, "h1", 0, 1.0, 0.0, 0.0),
        Edge("h1", 0, "h2", 0, 1.0, 0.0, 0.0),
        Edge("h1", 1, "h2", 0, 1.0, 0.0, 0.0),
        Edge("h2", 0, "out", 0, 1.0, 0.0, 0.0),
        Edge("h2", 1, "out", 0, 1.0, 0.0, 0.0),
    )

    selected = select_edges_by_effect(edges, state, edge_effect_quantile=0.9)

    assert {(edge.source_layer, edge.source_index, edge.target_layer) for edge in selected} == {
        ("in", 0, "h1"),
        ("h1", 0, "h2"),
        ("h2", 0, "out"),
    }
