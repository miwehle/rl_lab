import torch

from dqn.model import DQN
from reward_shaping.experiment_harness import load_q_net_checkpoint, save_q_net_checkpoint


def test_load_q_net_checkpoint_restores_saved_model_state_and_metadata(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    q_net = DQN(1, 2, hidden_size=4)
    for parameter in q_net.parameters():
        parameter.data.fill_(0.5)
    save_q_net_checkpoint(q_net, path, {"score": 253.0})

    restored = DQN(1, 2, hidden_size=4)
    metadata = load_q_net_checkpoint(restored, path, device=torch.device("cpu"))

    assert metadata == {"score": 253.0}
    for original, loaded in zip(q_net.parameters(), restored.parameters(), strict=True):
        assert torch.equal(original, loaded)
