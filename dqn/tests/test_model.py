import pytest
import torch

from dqn.model import DQN


def test_dqn_uses_hidden_size_for_both_hidden_layers():
    model = DQN(10, 4, hidden_size=64)

    output = model(torch.zeros(3, 10))

    assert output.shape == (3, 4)
    assert model.layer1.out_features == 64
    assert model.layer2.out_features == 64


def test_dqn_accepts_independent_hidden_layer_sizes():
    model = DQN(10, 4, hidden_sizes=(8, 6))

    output = model(torch.zeros(3, 10))

    assert output.shape == (3, 4)
    assert model.layer1.out_features == 8
    assert model.layer2.out_features == 6


def test_dqn_rejects_invalid_hidden_sizes():
    with pytest.raises(ValueError, match="exactly two"):
        DQN(10, 4, hidden_sizes=(64,))  # type: ignore[arg-type]
