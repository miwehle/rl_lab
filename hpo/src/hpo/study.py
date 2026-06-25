"""Shared Optuna orchestration for resumable HPO study series."""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hpo.lunar_lander.logging import log_call
from hpo.objective import (
    ObjectiveConfig,
    create_objective,
)
from hpo.study_reporting import RobustnessProgress, StudySeriesReporter


logger = logging.getLogger(__name__)

ProgressFn = Callable[..., None]
RobustnessProgressFn = Callable[[RobustnessProgress], None]
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
        def show_progress(study, *, target_trials):
            self.reporter.report_optimization(
                study,
                target_trials=target_trials,
                studies=[*self.studies, study],
                incumbent_params=self.incumbent_params,
            )

        def show_robustness(progress: RobustnessProgress):
            self.reporter.report_robustness_evaluation(
                study,
                studies=[*self.studies, study],
                incumbent_params=self.incumbent_params,
                progress=progress,
            )

        study = run_study(
            study_name=study_name,
            suggest_parameter_values=suggest_parameter_values,
            incumbent_params=self.incumbent_params,
            n_trials=n_trials,
            database_path=self.database_path(study_name),
            objective_cfg=self.objective_cfg,
            study_attrs=self.study_attrs,
            progress_fn=show_progress,
            sync_fn=self.sync_fn,
        )
        selected_params = select_robust_best(
            study=study,
            suggest_parameter_values=suggest_parameter_values,
            incumbent_params=self.incumbent_params,
            objective_cfg=self.objective_cfg,
            top_n=self.robust_candidates,
            extra_seeds=self.extra_seeds,
            progress_fn=show_robustness,
        )
        selected_score = study.user_attrs["robust_best_score"]
        if selected_score > self.incumbent_score:
            self.incumbent_params.update(selected_params)
            self.incumbent_score = selected_score

        study.set_user_attr("incumbent_params", self.incumbent_params)
        study.set_user_attr("incumbent_score", self.incumbent_score)
        if self.sync_fn is not None:
            self.sync_fn()
        show_progress(study, target_trials=n_trials)
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

    objective = create_objective(
        suggest_parameter_values=suggest_parameter_values,
        incumbent_params=incumbent_params,
        config=objective_cfg,
    )
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


def select_robust_best(
    *,
    study: Any,
    suggest_parameter_values: Any,
    incumbent_params: dict[str, Any],
    objective_cfg: ObjectiveConfig,
    top_n: int = 3,
    extra_seeds: Iterable[int] = (1001, 1002),
    progress_fn: RobustnessProgressFn | None = None,
) -> dict[str, Any]:
    """Re-check top candidates with extra seeds and return the best params."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    candidates = _top_complete_trials(study, top_n)
    extra_seeds = tuple(extra_seeds)
    if not candidates:
        raise ValueError("study has no complete trials")

    best_params = None
    best_mean = float("-inf")
    candidate_seed_scores = [
        [float(trial.value)]
        for trial in candidates
    ]

    def score_candidate(
        trial: Any,
        candidate_index: int,
    ) -> tuple[dict[str, Any], float]:
        scores = [float(trial.value)]

        for seed_index, seed_offset in enumerate(extra_seeds, start=1):
            if progress_fn is not None:
                progress_fn(
                    RobustnessProgress(
                        candidate_index=candidate_index,
                        candidate_count=len(candidates),
                        seed_index=seed_index,
                        seed_count=len(extra_seeds),
                        candidate_seed_scores=candidate_seed_scores,
                    )
                )
            objective = create_objective(
                suggest_parameter_values=suggest_parameter_values,
                incumbent_params=incumbent_params,
                config=replace(
                    objective_cfg,
                    training_seed=(
                        None
                        if objective_cfg.training_seed is None
                        else objective_cfg.training_seed + seed_offset
                    ),
                ),
            )
            fixed_trial = _FixedParamTrial(
                trial.params,
                number=trial.number,
                checkpoint_subdir="robustness",
                checkpoint_stem=f"trial_{trial.number:04d}_seed_{seed_offset}",
            )
            scores.append(objective(fixed_trial))
            candidate_seed_scores[candidate_index - 1] = list(scores)
            if progress_fn is not None:
                progress_fn(
                    RobustnessProgress(
                        candidate_index=candidate_index,
                        candidate_count=len(candidates),
                        seed_index=seed_index,
                        seed_count=len(extra_seeds),
                        candidate_seed_scores=candidate_seed_scores,
                    )
                )

        mean_score = sum(scores) / len(scores)
        return dict(trial.params), mean_score

    for candidate_index, trial in enumerate(candidates, start=1):
        params, mean_score = score_candidate(
            trial,
            candidate_index,
        )
        if mean_score > best_mean:
            best_mean = mean_score
            best_params = params

    selected_params = best_params or {}
    study.set_user_attr("robust_best_params", selected_params)
    study.set_user_attr("robust_best_score", best_mean)
    return selected_params


def _top_complete_trials(study: Any, top_n: int) -> list[Any]:
    trials = [
        trial for trial in study.trials
        if trial.state.name == "COMPLETE" and trial.value is not None
    ]
    trials.sort(key=lambda trial: trial.value, reverse=True)
    return trials[:top_n]


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


class _FixedParamTrial:
    """Stand-in for an Optuna Trial during robustness evaluation.

    It replays fixed suggested HP values, but gets its own trial number for
    checkpoint files, so robustness evaluation can use the same objective and
    checkpointing chain as normal training trials.
    """

    def __init__(
        self,
        params: dict[str, Any],
        *,
        number: int,
        checkpoint_subdir: str,
        checkpoint_stem: str,
    ) -> None:
        self.number = number
        self.checkpoint_subdir = checkpoint_subdir
        self.checkpoint_stem = checkpoint_stem
        self.params = params
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        return self.params[name]

    def suggest_float(self, name, low, high, *, log=False):
        return self.params[name]

    def suggest_int(self, name, low, high, *, log=False):
        return self.params[name]

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value
