"""Reward shaping workspace package."""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent / "src" / "reward_shaping"))

from reward_shaping.ground_side_thrust import RewardShapingEnv

__all__ = ["RewardShapingEnv"]
