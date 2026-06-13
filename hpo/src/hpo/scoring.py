"""Scoring helpers for HPO trial results."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowScore:
    mean: float
    start_episode: int
    end_episode: int


def best_window_mean(values: Sequence[float], window: int) -> WindowScore:
    """Return the best rolling mean and its 1-based episode range."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if not values:
        raise ValueError("values must not be empty")

    best_score = WindowScore(
        mean=sum(values[:window]) / min(window, len(values)),
        start_episode=1,
        end_episode=min(window, len(values)),
    )

    for start_index in range(1, max(len(values) - window + 1, 1)):
        end_index = start_index + window
        mean = sum(values[start_index:end_index]) / window
        if mean > best_score.mean:
            best_score = WindowScore(
                mean=mean,
                start_episode=start_index + 1,
                end_episode=end_index,
            )

    return best_score

