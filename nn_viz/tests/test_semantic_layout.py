import numpy as np
import torch

from dqn.model import DQN
from nn_viz.activations import ActivationRollouts
from nn_viz.layout import compute_semantic_layout


def test_compute_semantic_layout_sorts_h1_by_dominant_input_group_then_activity():
    q_net = DQN(10, 4, hidden_sizes=(4, 1))
    with torch.no_grad():
        q_net.layer1.weight.zero_()
        q_net.layer1.weight[0, 3] = 10.0
        q_net.layer1.weight[1, 0] = 10.0
        q_net.layer1.weight[2, 2] = 10.0
        q_net.layer1.weight[3, 4] = 10.0
        q_net.layer2.weight[:] = torch.ones((1, 4))
        q_net.layer3.weight[:] = torch.ones((4, 1))
    rollouts = ActivationRollouts(
        observations=np.ones((3, 10), dtype=np.float32),
        h1=np.array([[1.0, 0.0, 2.0, 1.0], [1.0, 3.0, 2.0, 1.0], [0.0, 3.0, 2.0, 1.0]], dtype=np.float32),
        h2=np.ones((3, 1), dtype=np.float32),
        q_values=np.zeros((3, 4), dtype=np.float32),
        actions=np.array([1, 1, 1], dtype=np.int64),
        rows=(),
    )

    layout = compute_semantic_layout(rollouts, q_net, top_edges_per_target=1)
    h1_order = [node.index for node in sorted((node for node in layout.nodes if node.layer == "h1"), key=lambda node: node.x)]

    assert h1_order == [2, 1, 0, 3]


def test_compute_semantic_layout_groups_h2_by_action_and_sorts_by_contribution():
    q_net = DQN(10, 4, hidden_sizes=(1, 4))
    with torch.no_grad():
        q_net.layer1.weight[:] = torch.ones((1, 10))
        q_net.layer2.weight[:] = torch.ones((4, 1))
        q_net.layer3.weight[:] = torch.tensor(
            [
                [0.1, 1.0, 0.1, 0.1],
                [0.1, 0.1, 6.0, 5.0],
                [0.1, 0.1, 0.1, 0.1],
                [4.0, 0.1, 0.1, 0.1],
            ]
        )
    rollouts = ActivationRollouts(
        observations=np.ones((2, 10), dtype=np.float32),
        h1=np.ones((2, 1), dtype=np.float32),
        h2=np.ones((2, 4), dtype=np.float32),
        q_values=np.zeros((2, 4), dtype=np.float32),
        actions=np.array([1, 1], dtype=np.int64),
        rows=(),
    )

    layout = compute_semantic_layout(rollouts, q_net, top_edges_per_target=1)
    h2_nodes = [node for node in layout.nodes if node.layer == "h2"]

    assert [(node.index, node.output_group) for node in h2_nodes] == [(2, 1), (3, 1), (1, 0), (0, 3)]
