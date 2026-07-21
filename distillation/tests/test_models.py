import pytest
import torch

from distillation.models import StudentDQN


def test_student_dqn_maps_observations_to_q_values():
    model = StudentDQN(10, 4, hidden_sizes=(64, 64))

    output = model(torch.zeros(3, 10))

    assert output.shape == (3, 4)


def test_student_dqn_rejects_invalid_hidden_sizes():
    with pytest.raises(ValueError, match="exactly two"):
        StudentDQN(10, 4, hidden_sizes=(64,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive"):
        StudentDQN(10, 4, hidden_sizes=(64, 0))
