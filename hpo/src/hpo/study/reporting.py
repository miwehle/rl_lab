"""Reporting contract for HPO study progress."""

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
    title: str = "HP Robustness Evaluation"
    step_label: str = "Seed"
    first_score_label: str = "Optimize trial"
    extra_score_label: str = "Extra seed"
    checkpoint_summaries: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class TrainingProgress:
    trial_number: int
    target_episodes: int
    episode_returns: list[float]
    episode_epsilons: list[float] | None = None
    episode_env_labels: list[str | None] | None = None
    trial_params: dict[str, Any] | None = None
    optimized_param_names: list[str] | None = None
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
    env_labels: list[str | None] | None = None
    trial_params: dict[str, Any] | None = None
    optimized_param_names: list[str] | None = None

    def plot_returns(self, episode_returns: list[float], **kwargs) -> None:
        env_indices = kwargs.get("env_indices")
        self.progress_fn(
            TrainingProgress(
                trial_number=self.trial_number,
                target_episodes=self.target_episodes,
                episode_returns=list(episode_returns),
                episode_epsilons=(list(kwargs["epsilons"]) if kwargs.get("epsilons") is not None else None),
                episode_env_labels=(
                    [
                        (
                            self.env_labels[index]
                            if self.env_labels is not None and 0 <= index < len(self.env_labels)
                            else None
                        )
                        for index in env_indices
                    ]
                    if env_indices is not None
                    else None
                ),
                trial_params=(dict(self.trial_params) if self.trial_params is not None else None),
                optimized_param_names=(
                    list(self.optimized_param_names) if self.optimized_param_names is not None else None
                ),
                checkpoint_window=self.checkpoint_window,
                checkpoint_min_score=self.checkpoint_min_score,
                best_checkpoint_score=(
                    self.best_checkpoint_score() if self.best_checkpoint_score is not None else None
                ),
            )
        )


class StudyReporter(Protocol):

    def set_incumbent_context(self, *, incumbent_params: dict[str, Any]) -> None: ...

    def report_optimization(self, study: Any, *, target_trials: int) -> None:
        """Report on all trials (overview)."""

    def report_robustness_evaluation(self, progress: RobustnessProgress) -> None: ...

    def report_training_progress(self, progress: TrainingProgress) -> None:
        """Report on one training (detail)."""
