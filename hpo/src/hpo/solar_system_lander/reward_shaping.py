"""Training-only reward shaping wrappers for SolarSystemLander."""

from collections.abc import Iterable

import gymnasium as gym


class GroundThrustPenaltyEnv(gym.Wrapper):
    """Penalize side-thrust while both lander legs touch ground and the body is awake."""

    def __init__(
        self, env: gym.Env, *, ground_thrust_penalty: float, side_thruster_actions: Iterable[int] = (1, 3)
    ) -> None:
        if ground_thrust_penalty < 0:
            raise ValueError("ground_thrust_penalty must be >= 0")
        super().__init__(env)
        self.ground_thrust_penalty = float(ground_thrust_penalty)
        self.side_thruster_actions = frozenset(side_thruster_actions)

    def step(self, action):
        should_penalize = self._is_ground_side_thrust(int(action))
        observation, reward, terminated, truncated, info = self.env.step(action)
        if should_penalize:
            reward = float(reward) - self.ground_thrust_penalty
        return observation, reward, terminated, truncated, info

    def _is_ground_side_thrust(self, action: int) -> bool:
        """Return whether action is side-thrust while both legs are grounded and the lander is awake."""
        if action not in self.side_thruster_actions:
            return False

        env = self.env.unwrapped
        legs = getattr(env, "legs", ())
        lander = getattr(env, "lander", None)
        if len(legs) < 2 or lander is None:
            return False

        both_legs_grounded = bool(legs[0].ground_contact and legs[1].ground_contact)
        lander_awake = bool(getattr(lander, "awake", False))
        return both_legs_grounded and lander_awake
