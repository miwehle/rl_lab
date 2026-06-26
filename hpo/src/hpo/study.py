"""Shared Optuna orchestration for resumable HPO study series."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hpo.lunar_lander.logging import log_call
from hpo.objective import (
    ObjectiveConfig,
    create_objective,
)
from hpo.robust_selection import select_robust_best
from hpo.study_reporting import (
    StudySeriesReporter,
    StudySeriesReporting,
    TrainingProgressFn,
)


logger = logging.getLogger(__name__)

ProgressFn = Callable[..., None]
DatabasePathFn = Callable[[str], str | Path]
SyncFn = Callable[[], None]


@dataclass(frozen=True)
class Baseline:
    """Starting point for a study series."""

    params: dict[str, Any]
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params))

    @classmethod
    def from_database(
        cls,
        database_path: str | Path,
        study_name: str,
    ) -> "Baseline":
        study = _load_study(
            study_name=study_name,
            storage=f"sqlite:///{Path(database_path)}",
        )
        return cls(
            params=study.user_attrs["incumbent_params"],
            score=study.user_attrs["incumbent_score"],
        )


@dataclass
class StudyRunner:
    """Run a study series and retain its incumbent and results."""

    database_path: DatabasePathFn
    objective_cfg: ObjectiveConfig
    baseline: Baseline
    reporter: StudySeriesReporter
    study_attrs: dict[str, Any] = field(default_factory=dict)
    robust_candidates: int = 3
    extra_seeds: tuple[int, ...] = (1001, 1002)
    sync_fn: SyncFn | None = None
    studies: list[Any] = field(default_factory=list, init=False)
    incumbent_params: dict[str, Any] = field(init=False)
    incumbent_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.incumbent_params = dict(self.baseline.params)
        self.incumbent_score = self.baseline.score

    def run(
        self,
        study_name: str,
        suggest_parameter_values: Any,
        n_trials: int,
    ) -> None:
        objective_cfg = _with_checkpoint_min_score(
            self.objective_cfg,
            self.incumbent_score,
        )
        objective_cfg = _with_training_progress(
            objective_cfg,
            self.reporter.report_training_progress,
        )
        reporting = StudySeriesReporting(
            reporter=self.reporter,
            previous_studies=self.studies,
            incumbent_params=self.incumbent_params,
        )

        study = run_study(
            study_name=study_name,
            suggest_parameter_values=suggest_parameter_values,
            incumbent_params=self.incumbent_params,
            n_trials=n_trials,
            database_path=self.database_path(study_name),
            objective_cfg=objective_cfg,
            study_attrs=self.study_attrs,
            progress_fn=reporting.report_optimization,
            sync_fn=self.sync_fn,
        )
        selected_params = select_robust_best(
            study=study,
            suggest_parameter_values=suggest_parameter_values,
            incumbent_params=self.incumbent_params,
            objective_cfg=objective_cfg,
            top_n=self.robust_candidates,
            extra_seeds=self.extra_seeds,
            progress_fn=reporting.report_robustness(study),
        )
        selected_score = study.user_attrs["robust_best_score"]
        if selected_score > self.incumbent_score:
            self.incumbent_params.update(selected_params)
            self.incumbent_score = selected_score

        study.set_user_attr("incumbent_params", self.incumbent_params)
        study.set_user_attr("incumbent_score", self.incumbent_score)
        if self.sync_fn is not None:
            self.sync_fn()
        reporting.report_optimization(study, target_trials=n_trials)
        self.studies.append(study)


@log_call
def run_study(
    *,
    study_name: str,
    suggest_parameter_values: Any,
    incumbent_params: dict[str, Any],
    n_trials: int,
    database_path: str | Path,
    objective_cfg: ObjectiveConfig,
    study_attrs: dict[str, Any] | None = None,
    progress_fn: ProgressFn | None = None,
    sync_fn: SyncFn | None = None,
) -> Any:
    """Create or load an Optuna study and run it to the target trial count.

    database_path: Path to the SQLite database.
    study_attrs: Study metadata to store and validate when resuming.
    n_trials: Target total number of finished trials.
    """
    logger.info("study: %s", study_name)

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    study = _create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{database_path}",
        load_if_exists=True,
    )
    _set_or_check_study_attrs(
        study,
        objective_cfg.study_attrs() | (study_attrs or {}),
    )

    objective = create_objective(
        suggest_parameter_values=suggest_parameter_values,
        incumbent_params=incumbent_params,
        config=objective_cfg,
    )
    if progress_fn is not None:
        progress_fn(study, target_trials=n_trials)

    while _finished_trial_count(study) < n_trials:
        logger.info("study.optimize")
        study.optimize(objective, n_trials=1)
        if sync_fn is not None:
            sync_fn()
        if progress_fn is not None:
            progress_fn(study, target_trials=n_trials)

    return study


def _with_checkpoint_min_score(
    objective_cfg: ObjectiveConfig,
    min_score: float,
) -> ObjectiveConfig:
    with_min_score = getattr(objective_cfg.hooks, "with_min_score", None)
    if with_min_score is None:
        return objective_cfg
    return replace(objective_cfg, hooks=with_min_score(min_score))


def _with_training_progress(
    objective_cfg: ObjectiveConfig,
    progress_fn: TrainingProgressFn | None,
) -> ObjectiveConfig:
    with_progress = getattr(objective_cfg.hooks, "with_training_progress", None)
    if with_progress is None:
        return objective_cfg
    return replace(objective_cfg, hooks=with_progress(progress_fn))


@log_call
def _create_study(**kwargs) -> Any:
    import optuna

    return optuna.create_study(**kwargs)


def _load_study(**kwargs) -> Any:
    import optuna

    return optuna.load_study(**kwargs)


def _set_or_check_study_attrs(study: Any, attrs: dict[str, Any]) -> None:
    """Prevent resuming a study with a different configuration."""
    for name, value in attrs.items():
        if name in study.user_attrs and study.user_attrs[name] != value:
            raise ValueError(f"study {name} does not match current configuration")
        study.set_user_attr(name, value)


def _finished_trial_count(study: Any) -> int:
    return sum(
        trial.state.name in {"COMPLETE", "PRUNED"}
        for trial in study.trials
    )

