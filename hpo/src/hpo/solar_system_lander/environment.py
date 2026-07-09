"""Mixed-world LunarLander environments for SolarSystemLander HPO."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
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


class World(StrEnum):
    MOON = "moon"
    MERCURY = "mercury"
    MARS = "mars"
    EARTH = "earth"
    VENUS = "venus"


WORLDS = (
    WorldConfig(World.MOON, -1.65, (0.0, 0.0), (0.0, 0.0)),
    WorldConfig(World.MERCURY, -3.7, (0.0, 0.0), (0.0, 0.0)),
    WorldConfig(World.MARS, -3.8, (0.0, 4.0), (0.0, 1.0)),
    WorldConfig(World.EARTH, -10.0, (5.0, 15.0), (0.0, 2.0)),
    WorldConfig(World.VENUS, -9.0, (15.0, 20.0), (0.0, 2.0)),
)
DEFAULT_WORLD_MIX = {world.name: 1 for world in WORLDS}


def acceleration_vector(previous_observation: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Return clipped velocity delta from two LunarLander observations."""
    previous_velocity = previous_observation[2:4]
    velocity = observation[2:4]
    return np.clip(velocity - previous_velocity, -1.0, 1.0).astype(np.float32)


class EnvWrapper(gym.Wrapper):
    """Apply episodic weather and optionally expose world parameters."""

    def __init__(self, env, world: WorldConfig, observation_mode: str) -> None:
        super().__init__(env)
        if observation_mode not in {"8d", "9d", "10d", "11d"}:
            raise ValueError("observation_mode must be '8d', '9d', '10d', or '11d'")

        self.world = world
        self.observation_mode = observation_mode
        self._weather_rng = np.random.default_rng()
        self._weather = (0.0, 0.0)
        self._previous_observation: np.ndarray | None = None
        self._acceleration = np.zeros(2, dtype=np.float32)

        if observation_mode in {"9d", "10d", "11d"}:
            base = env.observation_space
            extra_low, extra_high = _extra_bounds(observation_mode)
            self.observation_space = Box(
                low=np.concatenate((base.low, extra_low)),
                high=np.concatenate((base.high, extra_high)),
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
        self._previous_observation = observation
        self._acceleration = np.zeros(2, dtype=np.float32)
        return self._observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._previous_observation is not None:
            self._acceleration = acceleration_vector(self._previous_observation, observation)
        self._previous_observation = observation
        return self._observation(observation), reward, terminated, truncated, info

    def _observation(self, observation):
        if self.observation_mode == "8d":
            return observation
        if self.observation_mode == "10d":
            # append seat-of-the-pants input to observation for 10d Elise
            return np.concatenate((observation, self._acceleration)).astype(np.float32, copy=False)
        wind, turbulence = self._weather
        values = [self.world.gravity / 12]
        if self.observation_mode == "11d":
            values.extend([wind / 20, turbulence / 2])
        world = np.array(values, dtype=np.float32)
        return np.concatenate((observation, world)).astype(np.float32, copy=False)


class EnvFactory:
    def __init__(
        self,
        observation_mode: str,
        *,
        world_mix: Mapping[str, int],
        training_env_wrapper: Callable[[gym.Env], gym.Env] | None = None,
    ) -> None:
        """Create env factories.

        training_env_wrapper wraps only envs built by make_training_env(), for example for reward shaping.
        """
        if observation_mode not in {"8d", "9d", "10d", "11d"}:
            raise ValueError("observation_mode must be '8d', '9d', '10d', or '11d'")
        worlds = _worlds_from_mix(world_mix)
        self.observation_mode = observation_mode
        self.worlds = worlds
        self.training_env_wrapper = training_env_wrapper

    def make_training_env(self, num_envs: int) -> SyncVectorEnv:
        if num_envs % len(self.worlds):
            raise ValueError(f"num_envs must be divisible by {len(self.worlds)}")
        slots_per_world = num_envs // len(self.worlds)
        factories = [self._training_factory(world) for world in self.worlds for _ in range(slots_per_world)]
        return SyncVectorEnv(factories)

    def evaluation_envs(self) -> dict[str, Callable[[], Any]]:
        return {world.name: self._factory(world) for world in self.worlds}

    def make_env(self, world_name: str, *, render_mode: str | None = None):
        """Create one environment for the requested world."""
        for world in self.worlds:
            if world.name == world_name:
                return self._factory(world, render_mode=render_mode)()
        raise ValueError(f"unknown world: {world_name}")

    def metadata(self) -> dict[str, Any]:
        return {
            "observation_mode": self.observation_mode,
            "worlds": [
                {
                    **asdict(world),
                    "wind_power": list(world.wind_power),
                    "turbulence_power": list(world.turbulence_power),
                }
                for world in self.worlds
            ],
        }

    def _factory(self, world: WorldConfig, *, render_mode: str | None = None) -> Callable[[], Any]:
        return lambda: EnvWrapper(
            gym.make(
                "LunarLander-v3",
                gravity=world.gravity,
                enable_wind=world.wind_power[1] > 0 or world.turbulence_power[1] > 0,
                render_mode=render_mode,
            ),
            world,
            self.observation_mode,
        )

    def _training_factory(self, world: WorldConfig) -> Callable[[], Any]:
        """Return a per-world training factory that can apply the reward-shaping wrapper."""
        def make_env():
            env = self._factory(world)()
            if self.training_env_wrapper is not None:
                env = self.training_env_wrapper(env)
            return env

        return make_env


def _extra_bounds(observation_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if observation_mode == "9d":
        return (np.array([-1.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
    if observation_mode == "10d":
        return (np.array([-1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32))
    return (np.array([-1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 1.0], dtype=np.float32))


def _worlds_from_mix(world_mix: Mapping[str, int]) -> tuple[WorldConfig, ...]:
    if not world_mix:
        raise ValueError("world_mix must not be empty")
    worlds = {world.name: world for world in WORLDS}
    expanded = []
    for world_name, count in world_mix.items():
        if count < 1:
            raise ValueError(f"world_mix count must be >= 1: {world_name}")
        try:
            world = worlds[world_name]
        except KeyError:
            raise ValueError(f"unknown world: {world_name}") from None
        expanded.extend([world] * count)
    return tuple(expanded)
