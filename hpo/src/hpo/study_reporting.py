"""Reporting contract for HPO study-series progress."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RobustnessProgress:
    candidate_index: int
    candidate_count: int
    seed_index: int
    seed_count: int
    candidate_seed_scores: list[list[float]]


@dataclass(frozen=True)
class TrainingProgress:
    trial_number: int
    target_episodes: int
    episode_returns: list[float]
    episode_epsilons: list[float] | None = None
    checkpoint_window: int | None = None
    checkpoint_min_score: float | None = None
    best_checkpoint_score: float | None = None


TrainingProgressFn = Callable[[TrainingProgress], None]


@dataclass
class TrainingProgressPlotter:
    trial_number: int
    target_episodes: int
    progress_fn: TrainingProgressFn
    checkpoint_window: int | None = None
    checkpoint_min_score: float | None = None
    best_checkpoint_score: Callable[[], float | None] | None = None

    def plot_returns(self, episode_returns: list[float], **kwargs) -> None:
        self.progress_fn(
            TrainingProgress(
                trial_number=self.trial_number,
                target_episodes=self.target_episodes,
                episode_returns=list(episode_returns),
                episode_epsilons=(
                    list(kwargs["epsilons"])
                    if kwargs.get("epsilons") is not None
                    else None
                ),
                checkpoint_window=self.checkpoint_window,
                checkpoint_min_score=self.checkpoint_min_score,
                best_checkpoint_score=(
                    self.best_checkpoint_score()
                    if self.best_checkpoint_score is not None
                    else None
                ),
            )
        )


class StudySeriesReporter(Protocol):
    def set_study_series_context(
        self,
        *,
        studies: list[Any],
        incumbent_params: dict[str, Any],
    ) -> None: ...

    def report_optimization(
        self,
        study: Any,
        *,
        target_trials: int,
    ) -> None: ...

    def report_robustness_evaluation(
        self,
        progress: RobustnessProgress,
    ) -> None: ...

    def report_training_progress(
        self,
        progress: TrainingProgress,
    ) -> None: ...
