import numpy as np
import pytest

from nn_viz._edges import representative_edges
from nn_viz._rendering import NetworkState
from nn_viz.layout import Edge


def test_representative_edges_splits_budget_per_target_by_sign_share():
    state = NetworkState(
        inputs=np.array([10.0, 4.0, 1.0], dtype=np.float32),
        h1=np.array([2.0], dtype=np.float32),
        h2=np.array([3.0], dtype=np.float32),
        q_values=np.array([0.0], dtype=np.float32),
        action=0,
    )
    edges = (
        Edge("in", 0, "h1", 0, 1.0, 0.0, 0.0),
        Edge("in", 1, "h1", 0, 0.25, 0.0, 0.0),
        Edge("in", 0, "h1", 0, -0.5, 0.0, 0.0),
        Edge("in", 2, "h1", 0, -1.0, 0.0, 0.0),
        Edge("in", 1, "h1", 1, 1.0, 0.0, 0.0),
        Edge("in", 2, "h1", 1, 1.0, 0.0, 0.0),
    )

    selected = representative_edges(edges, state, max_edges_per_neuron=3)

    assert {(edge.source_index, edge.target_index, edge.weight) for edge in selected} == {
        (0, 0, 1.0),
        (1, 0, 0.25),
        (0, 0, -0.5),
        (1, 1, 1.0),
        (2, 1, 1.0),
    }


def test_representative_edges_does_not_force_irrelevant_signs():
    state = NetworkState(
        inputs=np.array([100.0, 1.0], dtype=np.float32),
        h1=np.array([], dtype=np.float32),
        h2=np.array([], dtype=np.float32),
        q_values=np.array([], dtype=np.float32),
        action=-1,
    )
    edges = (
        Edge("in", 0, "h1", 0, 1.0, 0.0, 0.0),
        Edge("in", 1, "h1", 0, -1.0, 0.0, 0.0),
    )

    selected = representative_edges(edges, state, max_edges_per_neuron=2)

    assert selected == (edges[0],)


def test_representative_edges_accepts_zero_budget():
    state = NetworkState(
        inputs=np.array([10.0, 1.0], dtype=np.float32),
        h1=np.array([], dtype=np.float32),
        h2=np.array([], dtype=np.float32),
        q_values=np.array([], dtype=np.float32),
        action=-1,
    )
    edges = (
        Edge("in", 0, "h1", 0, 1.0, 0.0, 0.0),
        Edge("in", 1, "h1", 0, 1.0, 0.0, 0.0),
    )

    assert representative_edges(edges, state, max_edges_per_neuron=0) == ()


def test_representative_edges_rejects_negative_budget():
    state = NetworkState(
        inputs=np.array([1.0], dtype=np.float32),
        h1=np.array([], dtype=np.float32),
        h2=np.array([], dtype=np.float32),
        q_values=np.array([], dtype=np.float32),
        action=-1,
    )
    edges = (Edge("in", 0, "h1", 0, 1.0, 0.0, 0.0),)

    with pytest.raises(ValueError, match="max_edges_per_neuron"):
        representative_edges(edges, state, max_edges_per_neuron=-1)
