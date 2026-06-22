"""Environment factory for LunarLander HPO."""

from collections.abc import Callable
from typing import Any

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv


class EnvFactory:
    def make_training_env(self, num_envs: int) -> SyncVectorEnv:
        return SyncVectorEnv([self._make_env for _ in range(num_envs)])

    def evaluation_envs(self) -> dict[str, Callable[[], Any]]:
        return {"lunar_lander": self._make_env}

    @staticmethod
    def _make_env():
        return gym.make("LunarLander-v3")
