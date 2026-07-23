import numpy as np

from dqn.model import DQN
from nn_viz.activations import RolloutSpec, collect_activations


class FakeEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self, *, seed=None):
        self.step_count = 0
        return np.full(10, seed or 0, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        observation = np.full(10, self.step_count, dtype=np.float32)
        return observation, 1.0, self.step_count == 2, False, {}

    def close(self):
        pass


class FakeEnvFactory:
    def make_env(self, world):
        return FakeEnv()


def test_collect_activations_records_one_row_per_environment_step():
    q_net = DQN(10, 4, hidden_sizes=(3, 2))

    rollouts = collect_activations(q_net, FakeEnvFactory(), [RolloutSpec("moon", seed=7)])

    assert rollouts.frame_count == 2
    assert rollouts.observations.shape == (2, 10)
    assert np.allclose(rollouts.observations[0], 7.0)
    assert rollouts.h1.shape == (2, 3)
    assert rollouts.h2.shape == (2, 2)
    assert rollouts.q_values.shape == (2, 4)
    assert rollouts.actions.shape == (2,)
    assert rollouts.rows[0]["world"] == "moon"
    assert rollouts.rows[0]["seed"] == 7
