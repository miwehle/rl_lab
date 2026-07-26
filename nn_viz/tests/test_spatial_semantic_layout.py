import numpy as np
import pytest
import torch

from dqn.model import DQN
from nn_viz.activations import ActivationRollouts
from nn_viz.layout import Node
from nn_viz.layout.spatial.semantic import compute_layout


def test_node_z_defaults_to_zero_for_2d_compatibility():
    assert Node("h1", 0, "H1-0", 1.0, 2.0, 0.3).z == 0.0


def test_spatial_semantic_layout_uses_fixed_input_and_output_anchors():
    q_net = DQN(10, 4, hidden_sizes=(1, 1))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer2.weight.zero_()
        q_net.layer3.weight.zero_()
    rollouts = _rollouts(h1_width=1, h2_width=1)

    layout = compute_layout(rollouts, q_net)
    nodes = {(node.layer, node.index): node for node in layout.nodes}

    assert _xyz(nodes[("in", 6)]) == (-1.0, 2.0, 0.0)
    assert _xyz(nodes[("in", 7)]) == (1.0, 2.0, 0.0)
    assert _xyz(nodes[("out", 1)]) == (-1.0, -1.0, 3.0)
    assert _xyz(nodes[("out", 3)]) == (1.0, -1.0, 3.0)
    assert _xyz(nodes[("out", 2)]) == (-1.0, 1.0, 3.0)
    assert _xyz(nodes[("out", 0)]) == (1.0, 1.0, 3.0)


def test_spatial_semantic_layout_places_h1_by_weighted_mean_of_input_anchors():
    q_net = DQN(10, 4, hidden_sizes=(2, 1))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer1.weight[0, 0] = 1.0
        q_net.layer1.weight[0, 2] = 1.0
        q_net.layer1.weight[1, 6] = 1.0
        q_net.layer1.weight[1, 7] = 3.0
        q_net.layer2.weight.zero_()
        q_net.layer3.weight.zero_()
    rollouts = _rollouts(h1_width=2, h2_width=1)

    layout = compute_layout(rollouts, q_net)
    h1 = {node.index: node for node in layout.nodes if node.layer == "h1"}

    assert np.allclose(_xyz(h1[0]), (-1.0, -1.0, 1.0))
    assert np.allclose(_xyz(h1[1]), (0.5, 2.0, 1.0))


def test_spatial_semantic_layout_places_h2_by_weighted_mean_of_h1_positions():
    q_net = DQN(10, 4, hidden_sizes=(2, 2))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer1.weight[0, 0] = 1.0
        q_net.layer1.weight[0, 2] = 1.0
        q_net.layer1.weight[1, 6] = 1.0
        q_net.layer1.weight[1, 7] = 3.0
        q_net.layer2.weight.zero_()
        q_net.layer2.weight[0, 0] = 1.0
        q_net.layer2.weight[0, 1] = 1.0
        q_net.layer2.weight[1, 1] = 1.0
        q_net.layer3.weight.zero_()
    rollouts = _rollouts(h1_width=2, h2_width=2)

    layout = compute_layout(rollouts, q_net)
    h2 = {node.index: node for node in layout.nodes if node.layer == "h2"}

    assert np.allclose(_xyz(h2[0]), (-0.25, 0.5, 2.0))
    assert np.allclose(_xyz(h2[1]), (0.5, 2.0, 2.0))


def test_spatial_semantic_layout_separates_overlapping_hidden_nodes_without_moving_layer_center():
    q_net = DQN(10, 4, hidden_sizes=(2, 1))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer1.weight[0, 0] = 1.0
        q_net.layer1.weight[0, 2] = 1.0
        q_net.layer1.weight[1, 0] = 1.0
        q_net.layer1.weight[1, 2] = 1.0
        q_net.layer2.weight.zero_()
        q_net.layer2.weight[0, 0] = 1.0
        q_net.layer3.weight.zero_()
    rollouts = _rollouts(h1_width=2, h2_width=1)

    layout = compute_layout(rollouts, q_net, min_node_distance=0.5)
    h1_positions = np.asarray([(node.x, node.y) for node in layout.nodes if node.layer == "h1"])

    assert np.allclose(np.mean(h1_positions, axis=0), (-1.0, -1.0))
    assert np.linalg.norm(h1_positions[0] - h1_positions[1]) >= 0.5


def test_spatial_semantic_layout_moves_stiffer_hidden_nodes_less_during_collision_relaxation():
    q_net = DQN(10, 4, hidden_sizes=(2, 1))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer1.weight[0, 0] = 10.0
        q_net.layer1.weight[0, 2] = 10.0
        q_net.layer1.weight[1, 0] = 1.0
        q_net.layer1.weight[1, 2] = 1.0
        q_net.layer2.weight.zero_()
        q_net.layer3.weight.zero_()
    rollouts = _rollouts(h1_width=2, h2_width=1)

    layout = compute_layout(rollouts, q_net, min_node_distance=0.5)
    h1 = {node.index: node for node in layout.nodes if node.layer == "h1"}
    desired_position = np.asarray((-1.0, -1.0))

    stiff_shift = np.linalg.norm(np.asarray((h1[0].x, h1[0].y)) - desired_position)
    weak_shift = np.linalg.norm(np.asarray((h1[1].x, h1[1].y)) - desired_position)
    assert stiff_shift < weak_shift


def test_spatial_semantic_layout_adds_edges_from_nonzero_weights():
    q_net = DQN(10, 4, hidden_sizes=(1, 2))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer1.weight[0, 0] = 1.0
        q_net.layer2.weight.zero_()
        q_net.layer2.weight[1, 0] = -2.0
        q_net.layer3.weight.zero_()
        q_net.layer3.weight[:, :] = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
    rollouts = _rollouts(h1_width=1, h2_width=2)

    layout = compute_layout(rollouts, q_net, edge_weight_quantile=0.0)
    edges = {
        (edge.source_layer, edge.source_index, edge.target_layer, edge.target_index): edge
        for edge in layout.edges
    }

    assert edges[("in", 0, "h1", 0)].weight == 1.0
    assert edges[("h1", 0, "h2", 1)].weight == -2.0
    output_edges = [
        edge for edge in layout.edges if edge.source_layer == "h2" and edge.target_layer == "out"
    ]
    assert len(output_edges) == 8
    assert ("h1", 0, "h2", 0) not in edges


def test_spatial_semantic_layout_limits_rendered_edges_by_top_k():
    q_net = DQN(10, 4, hidden_sizes=(1, 3))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer1.weight[0, 0] = 1.0
        q_net.layer1.weight[0, 1] = 3.0
        q_net.layer2.weight.zero_()
        q_net.layer2.weight[:, 0] = torch.tensor([1.0, 2.0, 3.0])
        q_net.layer3.weight.zero_()
        q_net.layer3.weight[1, :] = torch.tensor([8.0, 10.0, 9.0])
    rollouts = _rollouts(h1_width=1, h2_width=3)

    layout = compute_layout(
        rollouts,
        q_net,
        top_edges_per_target=1,
        output_edges_per_target=2,
        edge_weight_quantile=0.0,
    )

    input_edges = [edge for edge in layout.edges if edge.source_layer == "in"]
    output_edges = [edge for edge in layout.edges if edge.target_layer == "out"]
    assert [(edge.source_index, edge.target_index) for edge in input_edges] == [(1, 0)]
    assert {(edge.source_index, edge.target_index) for edge in output_edges} == {(1, 1), (2, 1)}


def test_spatial_semantic_layout_filters_top_k_candidates_by_layer_weight_quantile():
    q_net = DQN(10, 4, hidden_sizes=(1, 3))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer1.weight[0, 0] = 1.0
        q_net.layer1.weight[0, 1] = 3.0
        q_net.layer2.weight.zero_()
        q_net.layer3.weight.zero_()
        q_net.layer3.weight[1, :] = torch.tensor([1.0, 3.0, 2.0])
    rollouts = _rollouts(h1_width=1, h2_width=3)

    layout = compute_layout(
        rollouts,
        q_net,
        top_edges_per_target=2,
        output_edges_per_target=3,
        edge_weight_quantile=0.70,
    )

    assert {
        (edge.source_layer, edge.source_index, edge.target_layer, edge.target_index)
        for edge in layout.edges
    } == {("in", 1, "h1", 0), ("h2", 1, "out", 1)}


def test_spatial_semantic_layout_rejects_nonpositive_top_k():
    q_net = DQN(10, 4, hidden_sizes=(1, 1))
    rollouts = _rollouts(h1_width=1, h2_width=1)

    with pytest.raises(ValueError, match="top_edges_per_target"):
        compute_layout(rollouts, q_net, top_edges_per_target=0)
    with pytest.raises(ValueError, match="output_edges_per_target"):
        compute_layout(rollouts, q_net, output_edges_per_target=0)


def test_spatial_semantic_layout_rejects_negative_min_node_distance():
    q_net = DQN(10, 4, hidden_sizes=(1, 1))
    rollouts = _rollouts(h1_width=1, h2_width=1)

    with pytest.raises(ValueError, match="min_node_distance"):
        compute_layout(rollouts, q_net, min_node_distance=-0.1)


def test_spatial_semantic_layout_rejects_invalid_edge_weight_quantile():
    q_net = DQN(10, 4, hidden_sizes=(1, 1))
    rollouts = _rollouts(h1_width=1, h2_width=1)

    with pytest.raises(ValueError, match="edge_weight_quantile"):
        compute_layout(rollouts, q_net, edge_weight_quantile=-0.1)
    with pytest.raises(ValueError, match="edge_weight_quantile"):
        compute_layout(rollouts, q_net, edge_weight_quantile=1.1)


def _rollouts(*, h1_width: int, h2_width: int) -> ActivationRollouts:
    return ActivationRollouts(
        observations=np.ones((2, 10), dtype=np.float32),
        h1=np.ones((2, h1_width), dtype=np.float32),
        h2=np.ones((2, h2_width), dtype=np.float32),
        q_values=np.zeros((2, 4), dtype=np.float32),
        actions=np.array([1, 1], dtype=np.int64),
        rows=(),
    )


def _xyz(node: Node) -> tuple[float, float, float]:
    return node.x, node.y, node.z
