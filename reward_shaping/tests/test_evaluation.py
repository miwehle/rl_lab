import numpy as np
import pytest
import torch
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from reward_shaping.experiment_harness import historical_score, robust_score


class FixedActionQNet(torch.nn.Module):
    def __init__(self, action: int, n_actions: int = 4) -> None:
        super().__init__()
        self.action = action
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_values = torch.zeros((x.shape[0], self.n_actions), device=x.device)
        q_values[:, self.action] = 1.0
        return q_values


class Leg:
    ground_contact = True


class Lander:
    awake = True


class SeedRewardEnv(Env):
    action_space = Discrete(4)
    observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, world_offset: float) -> None:
        self.world_offset = world_offset
        self.seed_value = 10_000
        self.legs = [Leg(), Leg()]
        self.lander = Lander()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed_value = 10_000 if seed is None else seed
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        reward = self.world_offset + (self.seed_value - 10_000)
        return np.array([0.0], dtype=np.float32), reward, True, False, {}


def test_historical_score_uses_ten_seeds_per_world_and_means_world_scores() -> None:
    make_envs = {"earth": lambda: SeedRewardEnv(0.0), "mars": lambda: SeedRewardEnv(10.0)}

    result = historical_score(q_net=FixedActionQNet(1), make_envs=make_envs, device="cpu")

    assert result.measurement == "historical_score"
    assert result.episodes_per_world == 10
    assert result.world_scores == pytest.approx({"earth": 4.5, "mars": 14.5})
    assert result.score == pytest.approx(9.5)
    assert [row.seed for row in result.rows[:10]] == list(range(10_000, 10_010))
    assert sum(row.ground_side_thrust_steps for row in result.rows) == 20


def test_robust_score_uses_configured_number_of_seeds_per_world() -> None:
    result = robust_score(
        q_net=FixedActionQNet(0),
        make_envs={"earth": lambda: SeedRewardEnv(0.0)},
        episodes_per_world=3,
        device="cpu",
    )

    assert result.measurement == "robust_score"
    assert result.episodes_per_world == 3
    assert [row.seed for row in result.rows] == [10_000, 10_001, 10_002]
    assert [row.ground_side_thrust_steps for row in result.rows] == [0, 0, 0]


def test_historical_score_rejects_invalid_episode_count_through_common_evaluator() -> None:
    with pytest.raises(ValueError, match="episodes_per_world must be >= 1"):
        robust_score(
            q_net=FixedActionQNet(0),
            make_envs={"earth": lambda: SeedRewardEnv(0.0)},
            episodes_per_world=0,
        )
