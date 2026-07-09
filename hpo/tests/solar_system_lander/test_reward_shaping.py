from types import SimpleNamespace

import gymnasium as gym
import pytest

from hpo.solar_system_lander.reward_shaping import RewardShapingEnv


class FakeLanderEnv(gym.Env):
    def __init__(self, *, both_legs_grounded: bool, awake: bool) -> None:
        self.legs = [
            SimpleNamespace(ground_contact=both_legs_grounded),
            SimpleNamespace(ground_contact=both_legs_grounded),
        ]
        self.lander = SimpleNamespace(awake=awake)

    def step(self, action):
        return "observation", 10.0, False, False, {}


class FakeEnvWithoutLanderParts(gym.Env):
    def step(self, action):
        return "observation", 10.0, False, False, {}


class TestRewardShapingEnv:
    def test_step_penalizes_ground_side_thrust(self) -> None:
        env = RewardShapingEnv(FakeLanderEnv(both_legs_grounded=True, awake=True), ground_thrust_penalty=2.5)

        _observation, reward, _terminated, _truncated, _info = env.step(1)

        assert reward == 7.5

    def test_step_keeps_reward_without_ground_side_thrust(self) -> None:
        env = RewardShapingEnv(FakeLanderEnv(both_legs_grounded=True, awake=True), ground_thrust_penalty=2.5)

        _observation, reward, _terminated, _truncated, _info = env.step(2)

        assert reward == 10.0

    def test_step_keeps_reward_when_lander_parts_are_missing(self) -> None:
        env = RewardShapingEnv(FakeEnvWithoutLanderParts(), ground_thrust_penalty=2.5)

        _observation, reward, _terminated, _truncated, _info = env.step(1)

        assert reward == 10.0

    def test_init_rejects_negative_penalty(self) -> None:
        with pytest.raises(ValueError, match="ground_thrust_penalty must be >= 0"):
            RewardShapingEnv(FakeLanderEnv(both_legs_grounded=True, awake=True), ground_thrust_penalty=-0.1)
