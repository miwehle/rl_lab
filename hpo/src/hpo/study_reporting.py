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

    def plot_returns(self, episode_returns: list[float], **_kwargs) -> None:
        self.progress_fn(
            TrainingProgress(
                trial_number=self.trial_number,
                target_episodes=self.target_episodes,
                episode_returns=list(episode_returns),
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
    def report_optimization(
        self,
        study: Any,
        *,
        target_trials: int,
        studies: list[Any],
        incumbent_params: dict[str, Any],
    ) -> None: ...

    def report_robustness_evaluation(
        self,
        study: Any,
        *,
        studies: list[Any],
        incumbent_params: dict[str, Any],
        progress: RobustnessProgress,
    ) -> None: ...

    def report_training_progress(
        self,
        progress: TrainingProgress,
    ) -> None: ...


@dataclass
class StudySeriesReporting:
    reporter: StudySeriesReporter
    previous_studies: list[Any]
    incumbent_params: dict[str, Any]

    def report_optimization(self, study: Any, *, target_trials: int) -> None:
        self.reporter.report_optimization(
            study,
            target_trials=target_trials,
            studies=[*self.previous_studies, study],
            incumbent_params=self.incumbent_params,
        )

    def report_robustness(self, study: Any) -> Callable[[RobustnessProgress], None]:
        def report(progress: RobustnessProgress) -> None:
            self.reporter.report_robustness_evaluation(
                study,
                studies=[*self.previous_studies, study],
                incumbent_params=self.incumbent_params,
                progress=progress,
            )

        return report
