"""Reporting contract for HPO study-series progress."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RobustnessProgress:
    candidate_index: int
    candidate_count: int
    seed_index: int
    seed_count: int
    candidate_seed_scores: list[list[float]]


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
