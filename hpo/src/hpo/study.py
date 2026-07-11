"""Shared Optuna orchestration for resumable HPO studies."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hpo.lunar_lander.logging import log_call
from hpo.objective import ObjectiveConfig, create_objective
from hpo.evaluation.checkpoint_robustness import evaluate_checkpoint_robustness
from hpo.hyperparams import HP
from hpo.study_metadata import record_study_metadata
from hpo.study_reporting import StudyReporter, TrainingProgressFn

logger = logging.getLogger(__name__)

ProgressFn = Callable[..., None]
DatabasePathFn = Callable[[str], str | Path]
BackupFn = Callable[[], None]


@dataclass(frozen=True)
class Baseline:
    """Complete HP starting point for a study.

    At the start of a study sequence, score is typically omitted.
    For a resumed study, build the baseline with from_database; then score is included.

    Purpose of this class: HPs are required for training, the DQN, and related configuration.
    We provide them through the baseline. HPs optimized by Optuna are provided via
    trial.suggest_* and can be omitted in the baseline; else Optuna overrides.
    """

    params: dict[str, Any]
    score: float | None = None

    def __post_init__(self) -> None:
        params = dict(self.params)
        if "replay_memory_capacity" in params:
            params.setdefault(HP.REPLAY_MEMORY, params.pop("replay_memory_capacity"))
        object.__setattr__(self, "params", params)

    @classmethod
    def from_database(cls, database_path: str | Path, study_name: str) -> "Baseline":
        study = _load_study(study_name=study_name, storage=f"sqlite:///{Path(database_path)}")
        return cls.from_study(study)

    @classmethod
    def from_study(cls, study: Any) -> "Baseline":
        """Build a baseline from an Optuna study."""
        return cls(params=study.user_attrs["incumbent_params"], score=study.user_attrs["incumbent_score"])


@dataclass
class StudyRunner:
    """Run a study and keep its incumbent."""

    database_path: DatabasePathFn
    objective_cfg: ObjectiveConfig
    reporter: StudyReporter
    baseline: Baseline
    study_attrs: dict[str, Any] = field(default_factory=dict)
    robust_candidates: int = 3
    robust_eval_episodes: int = 50
    runtime_provider: str | None = None
    backup_fn: BackupFn | None = None
    incumbent_params: dict[str, Any] = field(init=False)
    incumbent_score: float | None = field(init=False)

    def __post_init__(self) -> None:
        self.incumbent_params = dict(self.baseline.params)
        self.incumbent_score = self.baseline.score

    def run(self, study_name: str, suggest_parameter_values: Any, n_trials: int) -> None:
        """Run or resume the named Optuna study in the configured storage.

        study_name: Name of the Optuna study inside the storage.
        """
        database_path = self.database_path(study_name)
        objective_cfg = self.objective_cfg
        if self.incumbent_score is not None:
            objective_cfg = _with_checkpoint_min_score(objective_cfg, self.incumbent_score)
        objective_cfg = _with_training_progress(objective_cfg, self.reporter.report_training_progress)

        study = _create_or_load_study(
            study_name=study_name,
            database_path=database_path,
            runtime_provider=self.runtime_provider,
            device=objective_cfg.device,
        )
        if _study_already_finished(study, n_trials):
            print("Study already finished.")
            return

        self.reporter.set_incumbent_context(incumbent_params=self.incumbent_params)

        run_study(
            study=study,
            suggest_parameter_values=suggest_parameter_values,
            incumbent_params=self.incumbent_params,
            n_trials=n_trials,
            objective_cfg=objective_cfg,
            study_attrs=self.study_attrs,
            progress_fn=self.reporter.report_optimization,
            backup_fn=self.backup_fn,
        )
        checkpoint_results = _evaluate_checkpoint_robustness(
            study=study,
            objective_cfg=objective_cfg,
            top_n=self.robust_candidates,
            eval_episodes=self.robust_eval_episodes,
            progress_fn=self.reporter.report_robustness_evaluation,
        )
        if checkpoint_results:
            winner = max(checkpoint_results, key=lambda result: result["robust_score"])
            selected_score = winner["robust_score"]
            if self.incumbent_score is None or selected_score > self.incumbent_score:
                self.incumbent_params.update(_trial_params(study, winner["trial_number"]))
                self.incumbent_score = selected_score

        study.set_user_attr("incumbent_params", self.incumbent_params)
        study.set_user_attr("incumbent_score", self.incumbent_score)
        if self.backup_fn is not None:
            self.backup_fn()
        self.reporter.set_incumbent_context(incumbent_params=self.incumbent_params)
        self.reporter.report_optimization(study, target_trials=n_trials)


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
    backup_fn: BackupFn | None = None,
) -> Any:
    """Run an Optuna study to the target trial count.

    study: Optuna study to optimize.
    study_attrs: Study metadata to store and validate when resuming.
    n_trials: Target total number of finished trials.
    backup_fn: Optional callback, usually for backing up DB/log from Colab storage to Google Drive.
    """
    logger.info("study: %s", getattr(study, "study_name", ""))

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    _set_or_check_study_attrs(study, objective_cfg.study_attrs() | (study_attrs or {}))

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
        if backup_fn is not None:
            backup_fn()
        if progress_fn is not None:
            progress_fn(study, target_trials=n_trials)

    return study


def _create_or_load_study(
    *, study_name: str, database_path: str | Path, runtime_provider: str | None = None, device: Any = None
) -> Any:
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    study = _create_study(
        study_name=study_name, direction="maximize", storage=f"sqlite:///{database_path}", load_if_exists=True
    )
    record_study_metadata(database_path, study_name, runtime_provider=runtime_provider, device=device)
    return study


def _with_checkpoint_min_score(objective_cfg: ObjectiveConfig, min_score: float) -> ObjectiveConfig:
    with_min_score = getattr(objective_cfg.hooks, "with_min_score", None)
    if with_min_score is None:
        return objective_cfg
    return replace(objective_cfg, hooks=with_min_score(min_score))


def _with_training_progress(
    objective_cfg: ObjectiveConfig, progress_fn: TrainingProgressFn | None
) -> ObjectiveConfig:
    with_progress = getattr(objective_cfg.hooks, "with_training_progress", None)
    if with_progress is None:
        return objective_cfg
    return replace(objective_cfg, hooks=with_progress(progress_fn))


def _evaluate_checkpoint_robustness(
    *,
    study: Any,
    objective_cfg: ObjectiveConfig,
    top_n: int,
    eval_episodes: int,
    progress_fn: ProgressFn | None,
) -> list[dict[str, Any]]:
    try:
        return evaluate_checkpoint_robustness(
            study=study,
            objective_cfg=objective_cfg,
            top_n=top_n,
            eval_episodes=eval_episodes,
            progress_fn=progress_fn,
        )
    except ValueError as error:
        if str(error) != "study has no evaluation checkpoints":
            raise
        study.set_user_attr("checkpoint_robustness", [])
        return []


def _trial_params(study: Any, trial_number: int) -> dict[str, Any]:
    for trial in study.trials:
        if trial.number == trial_number:
            return dict(trial.params)
    raise ValueError(f"study has no trial {trial_number}")


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
    return sum(trial.state.name in {"COMPLETE", "PRUNED"} for trial in study.trials)


def _study_already_finished(study: Any, n_trials: int) -> bool:
    return (
        n_trials >= 1
        and _finished_trial_count(study) >= n_trials
        and "checkpoint_robustness" in study.user_attrs
        and "incumbent_score" in study.user_attrs
    )
