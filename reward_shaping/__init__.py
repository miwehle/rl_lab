"""Reward shaping workspace package."""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent / "src" / "reward_shaping"))

from reward_shaping.ground_thrust_penalty import RewardShapingEnv, make_reward_shaping_vector_env

__all__ = ["RewardShapingEnv", "make_reward_shaping_vector_env"]
