import numpy as np
import pytest
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

from reward_shaping import RewardShapingEnv, make_reward_shaping_vector_env
from reward_shaping.ground_thrust_penalty import is_ground_side_thrust


class Leg:
    def __init__(self, ground_contact: bool) -> None:
        self.ground_contact = ground_contact


class Lander:
    def __init__(self, awake: bool) -> None:
        self.awake = awake


class FakeLanderEnv(Env):
    action_space = Discrete(4)
    observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, *, both_legs_grounded: bool, awake: bool, reward: float = 10.0) -> None:
        self.legs = [Leg(both_legs_grounded), Leg(both_legs_grounded)]
        self.lander = Lander(awake)
        self.reward = reward

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        return np.array([0.0], dtype=np.float32), self.reward, False, False, {}


class TestRewardShapingEnv:
    def test_step_penalizes_side_thrust_when_lander_is_awake_on_both_legs(self) -> None:
        env = RewardShapingEnv(FakeLanderEnv(both_legs_grounded=True, awake=True), ground_thrust_penalty=2.5)

        _, reward, _, _, _ = env.step(1)

        assert reward == pytest.approx(7.5)

    def test_step_keeps_reward_when_action_is_not_side_thrust(self) -> None:
        env = RewardShapingEnv(FakeLanderEnv(both_legs_grounded=True, awake=True), ground_thrust_penalty=2.5)

        _, reward, _, _, _ = env.step(0)

        assert reward == pytest.approx(10.0)

    def test_step_keeps_reward_when_only_one_leg_has_ground_contact(self) -> None:
        base_env = FakeLanderEnv(both_legs_grounded=True, awake=True)
        base_env.legs[1].ground_contact = False
        env = RewardShapingEnv(base_env, ground_thrust_penalty=2.5)

        _, reward, _, _, _ = env.step(3)

        assert reward == pytest.approx(10.0)

    def test_step_keeps_reward_when_lander_is_not_awake(self) -> None:
        env = RewardShapingEnv(FakeLanderEnv(both_legs_grounded=True, awake=False), ground_thrust_penalty=2.5)

        _, reward, _, _, _ = env.step(3)

        assert reward == pytest.approx(10.0)

    def test_init_rejects_negative_penalty(self) -> None:
        with pytest.raises(ValueError, match="ground_thrust_penalty must be >= 0"):
            RewardShapingEnv(FakeLanderEnv(both_legs_grounded=True, awake=True), ground_thrust_penalty=-0.1)


def test_is_ground_side_thrust_returns_false_when_lander_parts_are_missing() -> None:
    assert not is_ground_side_thrust(object(), 1)


class World:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeFactory:
    worlds = (World("earth"), World("mars"))

    def make_env(self, world_name: str):
        return FakeLanderEnv(both_legs_grounded=True, awake=True, reward=float(len(world_name)))


def test_make_reward_shaping_vector_env_wraps_one_slot_per_world() -> None:
    env = make_reward_shaping_vector_env(FakeFactory(), 2, ground_thrust_penalty=1.0)
    try:
        assert [sub_env.ground_thrust_penalty for sub_env in env.envs] == [1.0, 1.0]
        assert [sub_env.env.reward for sub_env in env.envs] == [5.0, 4.0]
    finally:
        env.close()


def test_make_reward_shaping_vector_env_requires_balanced_slots() -> None:
    with pytest.raises(ValueError, match="divisible by 2"):
        make_reward_shaping_vector_env(FakeFactory(), 3, ground_thrust_penalty=1.0)
