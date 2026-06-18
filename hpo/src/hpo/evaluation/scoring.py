"""Scoring helpers for HPO trial results."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, kw_only=True)
class ScoringConfig:
    alpha: float = 0.5
    quality_weight: float = 0.9
    quality_min: float = 200.0
    quality_target: float = 250.0
    eval_episodes: int = 20
    eval_seed: int = 10_000
    baseline_env_steps: float | None = None
    baseline_processed_samples: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 1 or not 0 <= self.quality_weight <= 1:
            raise ValueError("weights must be between 0 and 1")
        if self.quality_target <= self.quality_min or self.eval_episodes < 1:
            raise ValueError("invalid quality range or eval_episodes")
        if (self.baseline_env_steps is None) != (
            self.baseline_processed_samples is None
        ):
            raise ValueError("baseline effort values must both be set or both be None")

    def study_attrs(self) -> dict:
        attrs = asdict(self)
        attrs["eval_seeds"] = list(
            range(self.eval_seed, self.eval_seed + self.eval_episodes)
        )
        del attrs["eval_seed"]
        return {name: value for name, value in attrs.items() if value is not None}


def training_effort(
    *,
    env_steps: int,
    processed_samples: int,
    baseline_env_steps: float,
    baseline_processed_samples: float,
    alpha: float = 0.5,
) -> float:
    """Return training effort relative to a baseline."""
    if baseline_env_steps <= 0 or baseline_processed_samples <= 0:
        raise ValueError("baseline effort values must be > 0")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    return (
        alpha * env_steps / baseline_env_steps
        + (1 - alpha) * processed_samples / baseline_processed_samples
    )
