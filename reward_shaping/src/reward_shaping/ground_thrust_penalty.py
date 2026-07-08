"""Ground side-thrust reward shaping for lander environments."""

from collections.abc import Iterable

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv


class RewardShapingEnv(gym.Wrapper):
    """Penalize side-thrust while both lander legs touch ground and the body is still awake."""

    def __init__(
        self,
        env: gym.Env,
        *,
        ground_thrust_penalty: float,
        side_thruster_actions: Iterable[int] = (1, 3),
    ) -> None:
        if ground_thrust_penalty < 0:
            raise ValueError("ground_thrust_penalty must be >= 0")
        super().__init__(env)
        self.ground_thrust_penalty = float(ground_thrust_penalty)
        self.side_thruster_actions = frozenset(side_thruster_actions)

    def step(self, action):
        should_penalize = is_ground_side_thrust(self.env.unwrapped, int(action), self.side_thruster_actions)
        observation, reward, terminated, truncated, info = self.env.step(action)
        if should_penalize:
            reward = float(reward) - self.ground_thrust_penalty
        return observation, reward, terminated, truncated, info


def is_ground_side_thrust(env, action: int, side_thruster_actions: Iterable[int] = (1, 3)) -> bool:
    """Return whether action is side-thrust while both legs are grounded and the lander is awake."""
    if action not in side_thruster_actions:
        return False

    legs = getattr(env, "legs", ())
    lander = getattr(env, "lander", None)
    if len(legs) < 2 or lander is None:
        return False

    both_legs_grounded = bool(legs[0].ground_contact and legs[1].ground_contact)
    lander_awake = bool(getattr(lander, "awake", False))
    return both_legs_grounded and lander_awake


def make_reward_shaping_vector_env(factory, num_envs: int, *, ground_thrust_penalty: float) -> SyncVectorEnv:
    """Return vector envs whose sub-envs are wrapped with RewardShapingEnv."""
    if num_envs % len(factory.worlds):
        raise ValueError(f"num_envs must be divisible by {len(factory.worlds)}")

    slots_per_world = num_envs // len(factory.worlds)
    factories = [
        lambda world=world: RewardShapingEnv(
            factory.make_env(world.name), ground_thrust_penalty=ground_thrust_penalty
        )
        for world in factory.worlds
        for _ in range(slots_per_world)
    ]
    return SyncVectorEnv(factories)
