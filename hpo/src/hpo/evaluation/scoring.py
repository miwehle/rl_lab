"""Scoring helpers for HPO trial results."""


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
