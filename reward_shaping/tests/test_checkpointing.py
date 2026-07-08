import torch
import pytest

from dqn.model import DQN
from reward_shaping.experiment_harness import (
    load_q_net_checkpoint,
    save_q_net_checkpoint,
    vector_training_config_from_checkpoint,
)


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


def test_vector_training_config_from_checkpoint_uses_metadata_with_overrides(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    q_net = DQN(1, 2, hidden_size=4)
    save_q_net_checkpoint(
        q_net,
        path,
        {
            "training_config": {
                "num_episodes": 2000,
                "eps_start": 0.5,
                "eps_end": 0.044,
                "eps_decay": 38_793,
                "learning_rate": 0.0006229370728793535,
                "batch_size": 512,
                "gamma": 0.995,
                "tau": 0.002,
                "learning_starts": 2500,
                "optimize_every": 2,
                "hidden_size": 128,
            }
        },
    )

    config = vector_training_config_from_checkpoint(path, num_episodes=500, eps_start=0.1)

    assert config.num_episodes == 500
    assert config.eps_start == pytest.approx(0.1)
    assert config.eps_end == pytest.approx(0.044)
    assert config.eps_decay == 38_793
    assert config.hidden_size == 128
