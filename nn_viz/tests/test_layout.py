import numpy as np
import torch

from dqn.model import DQN
from nn_viz.activations import ActivationRollouts
from nn_viz.layout import compute_activity_layout


def test_compute_activity_layout_uses_activated_contributions_not_weights_only():
    q_net = DQN(10, 4, hidden_sizes=(2, 2))
    with torch.no_grad():
        q_net.layer2.weight[:] = torch.tensor([[10.0, 1.0], [1.0, 10.0]])
        q_net.layer3.weight[:] = torch.tensor(
            [
                [100.0, 1.0],
                [1.0, 5.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )
    rollouts = ActivationRollouts(
        observations=np.ones((2, 10), dtype=np.float32),
        h1=np.array([[0.0, 2.0], [0.0, 3.0]], dtype=np.float32),
        h2=np.array([[0.0, 4.0], [0.0, 5.0]], dtype=np.float32),
        q_values=np.zeros((2, 4), dtype=np.float32),
        actions=np.array([1, 1], dtype=np.int64),
        rows=(),
    )

    layout = compute_activity_layout(rollouts, q_net, top_edges_per_target=1)

    h2_nodes = {node.index: node for node in layout.nodes if node.layer == "h2"}
    assert h2_nodes[1].x == 0.0
    assert h2_nodes[0].x != 0.0
    assert any(
        edge.source_layer == "h2"
        and edge.source_index == 1
        and edge.target_layer == "out"
        and edge.target_index == 1
        for edge in layout.edges
    )


def test_compute_activity_layout_prefers_action_specific_output_edges():
    q_net = DQN(10, 4, hidden_sizes=(1, 2))
    with torch.no_grad():
        q_net.layer2.weight[:] = torch.tensor([[1.0], [1.0]])
        q_net.layer3.weight[:] = torch.tensor(
            [
                [10.0, 1.0],
                [10.0, 5.0],
                [10.0, 1.0],
                [10.0, 1.0],
            ]
        )
    rollouts = ActivationRollouts(
        observations=np.ones((2, 10), dtype=np.float32),
        h1=np.ones((2, 1), dtype=np.float32),
        h2=np.ones((2, 2), dtype=np.float32),
        q_values=np.zeros((2, 4), dtype=np.float32),
        actions=np.array([1, 1], dtype=np.int64),
        rows=(),
    )

    layout = compute_activity_layout(rollouts, q_net, top_edges_per_target=1)

    output_edges = [edge for edge in layout.edges if edge.target_layer == "out"]
    assert any(edge.target_index == 1 and edge.source_index == 1 for edge in output_edges)
    assert not any(edge.target_index == 1 and edge.source_index == 0 for edge in output_edges)


def test_compute_activity_layout_can_use_more_output_edges_than_hidden_edges():
    q_net = DQN(10, 4, hidden_sizes=(1, 6))
    with torch.no_grad():
        q_net.layer2.weight[:] = torch.ones((6, 1))
        q_net.layer3.weight[:] = torch.full((4, 6), 0.1)
        q_net.layer3.weight[0, [0, 1]] = torch.tensor([5.0, 4.0])
        q_net.layer3.weight[1, [2, 3]] = torch.tensor([5.0, 4.0])
        q_net.layer3.weight[2, [4, 5]] = torch.tensor([5.0, 4.0])
        q_net.layer3.weight[3, [1, 3]] = torch.tensor([3.0, 2.0])
    rollouts = ActivationRollouts(
        observations=np.ones((2, 10), dtype=np.float32),
        h1=np.ones((2, 1), dtype=np.float32),
        h2=np.ones((2, 6), dtype=np.float32),
        q_values=np.zeros((2, 4), dtype=np.float32),
        actions=np.array([1, 1], dtype=np.int64),
        rows=(),
    )

    layout = compute_activity_layout(rollouts, q_net, top_edges_per_target=1, output_edges_per_target=2)

    output_edges = [edge for edge in layout.edges if edge.target_layer == "out"]
    hidden_edges = [edge for edge in layout.edges if edge.target_layer == "h2"]
    assert len(output_edges) == 8
    assert len(hidden_edges) == 6


def test_compute_activity_layout_adds_observation_input_edges():
    q_net = DQN(10, 4, hidden_sizes=(2, 1))
    with torch.no_grad():
        q_net.layer1.weight[:] = torch.zeros((2, 10))
        q_net.layer1.weight[0, 3] = 1.0
        q_net.layer1.weight[0, 9] = 2.0
        q_net.layer1.weight[1, 0] = 5.0
        q_net.layer2.weight[:] = torch.ones((1, 2))
        q_net.layer3.weight[:] = torch.ones((4, 1))
    rollouts = ActivationRollouts(
        observations=np.array([[1, 1, 1, 2, 1, 1, 1, 1, 1, 3]], dtype=np.float32),
        h1=np.ones((1, 2), dtype=np.float32),
        h2=np.ones((1, 1), dtype=np.float32),
        q_values=np.zeros((1, 4), dtype=np.float32),
        actions=np.array([1], dtype=np.int64),
        rows=(),
    )

    layout = compute_activity_layout(rollouts, q_net, top_edges_per_target=2)

    input_nodes = [node for node in layout.nodes if node.layer == "in"]
    input_edges = [edge for edge in layout.edges if edge.source_layer == "in" and edge.target_layer == "h1"]
    assert [node.label for node in input_nodes] == ["x", "y", "vx", "vy", "ang", "vang", "ftl", "ftr", "ax", "ay"]
    assert {(edge.source_index, edge.target_index) for edge in input_edges} == {(9, 0), (3, 0), (0, 1)}
