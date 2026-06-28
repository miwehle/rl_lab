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
    TrainingProgressFn,
)


logger = logging.getLogger(__name__)

ProgressFn = Callable[..., None]
DatabasePathFn = Callable[[str], str | Path]
SyncFn = Callable[[], None]


@dataclass(frozen=True)
class Incumbent:
    """Best known params and score that may seed a study series."""

    params: dict[str, Any]
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", dict(self.params))

    @classmethod
    def from_database(
        cls,
        database_path: str | Path,
        study_name: str,
    ) -> "Incumbent":
        study = _load_study(
            study_name=study_name,
            storage=f"sqlite:///{Path(database_path)}",
        )
        return cls.from_study(study)

    @classmethod
    def from_study(cls, study: Any) -> "Incumbent":
        return cls(
            params=study.user_attrs["incumbent_params"],
            score=study.user_attrs["incumbent_score"],
        )


@dataclass
class StudyRunner:
    """Run a study series and retain its incumbent and results."""

    database_path: DatabasePathFn
    objective_cfg: ObjectiveConfig
    reporter: StudySeriesReporter
    baseline: Incumbent | None = None
    study_attrs: dict[str, Any] = field(default_factory=dict)
    robust_candidates: int = 3
    extra_seeds: tuple[int, ...] = (1001, 1002)
    sync_fn: SyncFn | None = None
    studies: list[Any] = field(default_factory=list, init=False)
    incumbent_params: dict[str, Any] = field(init=False)
    incumbent_score: float | None = field(init=False)

    def __post_init__(self) -> None:
        if self.baseline is None:
            self.incumbent_params = {}
            self.incumbent_score = None
            return
        self.incumbent_params = dict(self.baseline.params)
        self.incumbent_score = self.baseline.score

    def run(
        self,
        study_name: str,
        suggest_parameter_values: Any,
        n_trials: int,
    ) -> None:
        objective_cfg = self.objective_cfg
        if self.incumbent_score is not None:
            objective_cfg = _with_checkpoint_min_score(
                objective_cfg,
                self.incumbent_score,
            )
        objective_cfg = _with_training_progress(
            objective_cfg,
            self.reporter.report_training_progress,
        )

        study = _create_or_load_study(
            study_name=study_name,
            database_path=self.database_path(study_name),
        )
        if _study_already_finished(study, n_trials):
            print("Study already finished.")
            return

        self.reporter.set_study_series_context(
            studies=[*self.studies, study],
            incumbent_params=self.incumbent_params,
        )

        run_study(
            study=study,
            suggest_parameter_values=suggest_parameter_values,
            incumbent_params=self.incumbent_params,
            n_trials=n_trials,
            objective_cfg=objective_cfg,
            study_attrs=self.study_attrs,
            progress_fn=self.reporter.report_optimization,
            sync_fn=self.sync_fn,
        )
        selected_params = select_robust_best(
            study=study,
            suggest_parameter_values=suggest_parameter_values,
            incumbent_params=self.incumbent_params,
            objective_cfg=objective_cfg,
            top_n=self.robust_candidates,
            extra_seeds=self.extra_seeds,
            progress_fn=self.reporter.report_robustness_evaluation,
        )
        selected_score = study.user_attrs["robust_best_score"]
        if self.incumbent_score is None or selected_score > self.incumbent_score:
            self.incumbent_params.update(selected_params)
            self.incumbent_score = selected_score

        study.set_user_attr("incumbent_params", self.incumbent_params)
        study.set_user_attr("incumbent_score", self.incumbent_score)
        if self.sync_fn is not None:
            self.sync_fn()
        self.reporter.set_study_series_context(
            studies=[*self.studies, study],
            incumbent_params=self.incumbent_params,
        )
        self.reporter.report_optimization(study, target_trials=n_trials)
        self.studies.append(study)


@log_call
def run_study(
    *,
    study: Any,
    suggest_parameter_values: Any,
    incumbent_params: dict[str, Any],
    n_trials: int,
    objective_cfg: ObjectiveConfig,
    study_attrs: dict[str, Any] | None = None,
    progress_fn: ProgressFn | None = None,
    sync_fn: SyncFn | None = None,
) -> Any:
    """Run an Optuna study to the target trial count.

    study_attrs: Study metadata to store and validate when resuming.
    n_trials: Target total number of finished trials.
    """
    logger.info("study: %s", getattr(study, "study_name", ""))

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

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


def _create_or_load_study(
    *,
    study_name: str,
    database_path: str | Path,
) -> Any:
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return _create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{database_path}",
        load_if_exists=True,
    )


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


def _study_already_finished(study: Any, n_trials: int) -> bool:
    return (
        n_trials >= 1
        and _finished_trial_count(study) >= n_trials
        and "robust_best_score" in study.user_attrs
        and "incumbent_score" in study.user_attrs
    )

