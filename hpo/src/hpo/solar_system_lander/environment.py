"""Mixed-world LunarLander environments for SolarSystemLander HPO."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.vector import SyncVectorEnv
import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    name: str
    gravity: float
    wind_power: tuple[float, float]
    turbulence_power: tuple[float, float]


WORLDS = (
    WorldConfig("moon", -1.65, (0.0, 0.0), (0.0, 0.0)),
    WorldConfig("mercury", -3.7, (0.0, 0.0), (0.0, 0.0)),
    WorldConfig("mars", -3.8, (0.0, 4.0), (0.0, 1.0)),
    WorldConfig("earth", -10.0, (5.0, 15.0), (0.0, 2.0)),
    WorldConfig("venus", -9.0, (15.0, 20.0), (0.0, 2.0)),
)


class EnvWrapper(gym.Wrapper):
    """Apply episodic weather and optionally expose world parameters."""

    def __init__(self, env, world: WorldConfig, observation_mode: str) -> None:
        super().__init__(env)
        if observation_mode not in {"8d", "11d"}:
            raise ValueError("observation_mode must be '8d' or '11d'")

        self.world = world
        self.observation_mode = observation_mode
        self._weather_rng = np.random.default_rng()
        self._weather = (0.0, 0.0)

        if observation_mode == "11d":
            base = env.observation_space
            self.observation_space = Box(
                low=np.concatenate((
                    base.low,
                    np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                )),
                high=np.concatenate((
                    base.high,
                    np.array([0.0, 1.0, 1.0], dtype=np.float32),
                )),
                dtype=np.float32,
            )

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._weather_rng = np.random.default_rng(seed)
        wind = self._weather_rng.uniform(*self.world.wind_power)
        turbulence = self._weather_rng.uniform(*self.world.turbulence_power)
        self._weather = (float(wind), float(turbulence))

        base = self.env.unwrapped
        base.enable_wind = wind > 0 or turbulence > 0
        base.wind_power = float(wind)
        base.turbulence_power = float(turbulence)
        observation, info = self.env.reset(seed=seed, options=options)
        return self._observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(observation), reward, terminated, truncated, info

    def _observation(self, observation):
        if self.observation_mode == "8d":
            return observation
        wind, turbulence = self._weather
        world = np.array(
            [self.world.gravity / 12, wind / 20, turbulence / 2],
            dtype=np.float32,
        )
        return np.concatenate((observation, world)).astype(np.float32, copy=False)


class EnvFactory:
    def __init__(self, observation_mode: str) -> None:
        if observation_mode not in {"8d", "11d"}:
            raise ValueError("observation_mode must be '8d' or '11d'")
        self.observation_mode = observation_mode

    def make_training_env(self, num_envs: int) -> SyncVectorEnv:
        if num_envs % len(WORLDS):
            raise ValueError(f"num_envs must be divisible by {len(WORLDS)}")
        slots_per_world = num_envs // len(WORLDS)
        factories = [
            self._factory(world)
            for world in WORLDS
            for _ in range(slots_per_world)
        ]
        return SyncVectorEnv(factories)

    def evaluation_envs(self) -> dict[str, Callable[[], Any]]:
        return {world.name: self._factory(world) for world in WORLDS}

    def metadata(self) -> dict[str, Any]:
        return {
            "observation_mode": self.observation_mode,
            "worlds": [
                {
                    **asdict(world),
                    "wind_power": list(world.wind_power),
                    "turbulence_power": list(world.turbulence_power),
                }
                for world in WORLDS
            ],
        }

    def _factory(self, world: WorldConfig) -> Callable[[], Any]:
        return lambda: EnvWrapper(
            gym.make(
                "LunarLander-v3",
                gravity=world.gravity,
                enable_wind=world.wind_power[1] > 0
                or world.turbulence_power[1] > 0,
            ),
            world,
            self.observation_mode,
        )
