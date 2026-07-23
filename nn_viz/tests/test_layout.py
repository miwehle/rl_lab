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
