"""Shared Optuna study orchestration for HPO tasks."""

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hpo.evaluation.reporting import (
    finished_trial_count,
    show_lander_live_progress,
    show_robustness_progress,
    show_study_progress,
)
from hpo.evaluation.scoring import ScoringConfig
from hpo.lunar_lander.logging import log_call
from hpo.objective import EnvironmentFactory, TrialConfig, create_objective


logger = logging.getLogger(__name__)

ProgressFn = Callable[..., None]
StoragePathFn = Callable[[str], str | Path]
SyncFn = Callable[[], None]


@dataclass
class StudyRunner:
    """Run a study series and retain its incumbent and results."""

    storage_path: StoragePathFn
    environment_factory: EnvironmentFactory
    trial_cfg: TrialConfig
    incumbent_params: dict[str, Any]
    study_attrs: dict[str, Any] = field(default_factory=dict)
    robust_candidates: int = 3
    extra_seeds: tuple[int, ...] = (1001, 1002)
    sync_fn: SyncFn | None = None
    studies: list[Any] = field(default_factory=list, init=False)
    incumbent_score: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.incumbent_params = dict(self.incumbent_params)

    def run(
        self,
        study_name: str,
        search_space: Any,
        n_trials: int,
        scoring_cfg: ScoringConfig,
        *,
        robust: bool = True,
    ) -> None:
        def show_progress(study, *, target_trials):
            show_lander_live_progress(
                study,
                target_trials=target_trials,
                lander_studies=[*self.studies, study],
            )

        def show_robustness(**kwargs):
            show_robustness_progress(
                study,
                lander_studies=[*self.studies, study],
                **kwargs,
            )

        study = run_study(
            study_name=study_name,
            search_space=search_space,
            incumbent_params=self.incumbent_params,
            n_trials=n_trials,
            storage_path=self.storage_path(study_name),
            environment_factory=self.environment_factory,
            trial_cfg=self.trial_cfg,
            scoring_cfg=scoring_cfg,
            study_attrs=self.study_attrs,
            progress_fn=show_progress,
            sync_fn=self.sync_fn,
        )
        selected_params = (
            select_robust_best(
                study=study,
                search_space=search_space,
                incumbent_params=self.incumbent_params,
                environment_factory=self.environment_factory,
                trial_cfg=self.trial_cfg,
                scoring_cfg=scoring_cfg,
                base_seed=self.trial_cfg.seed,
                top_n=self.robust_candidates,
                extra_seeds=self.extra_seeds,
                progress_fn=show_robustness,
            )
            if robust
            else {}
        )
        selected_score = study.user_attrs["robust_best_objective_score"]
        if self.incumbent_score is None or selected_score > self.incumbent_score:
            self.incumbent_params.update(selected_params)
            self.incumbent_score = selected_score
        else:
            incumbent = self.studies[-1]
            retained = {
                name: self.incumbent_params[name]
                for name in selected_params
            }
            _set_study_user_attr(study, "robust_best_params", retained)
            for name in ("objective_score", "gym_score", "training_effort"):
                attr = f"robust_best_{name}"
                _set_study_user_attr(study, attr, incumbent.user_attrs[attr])
        if robust and self.sync_fn is not None:
            self.sync_fn()
        show_progress(study, target_trials=n_trials)
        self.studies.append(study)


@log_call
def run_study(
    *,
    study_name: str,
    search_space: Any,
    incumbent_params: dict[str, Any],
    n_trials: int,
    storage_path: str | Path,
    environment_factory: EnvironmentFactory,
    trial_cfg: TrialConfig = TrialConfig(),
    scoring_cfg: ScoringConfig = ScoringConfig(),
    study_attrs: dict[str, Any] | None = None,
    progress_fn: ProgressFn | None = show_study_progress,
    sync_fn: SyncFn | None = None,
) -> Any:
    """Create or load an Optuna study and run it to the target trial count."""
    logger.info("study: %s", study_name)

    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    objective = create_objective(
        search_space=search_space,
        incumbent_params=incumbent_params,
        environment_factory=environment_factory,
        trial_cfg=trial_cfg,
        scoring_cfg=scoring_cfg,
    )
    study = _create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )
    _set_or_check_study_attrs(
        study,
        scoring_cfg.study_attrs() | (study_attrs or {}),
    )
    if progress_fn is not None:
        progress_fn(study, target_trials=n_trials)

    while finished_trial_count(study) < n_trials:
        logger.info("study.optimize")
        study.optimize(objective, n_trials=1)
        if sync_fn is not None:
            sync_fn()
        if progress_fn is not None:
            progress_fn(study, target_trials=n_trials)

    if scoring_cfg.baseline_env_steps is None:
        def save(name, value):
            _set_study_user_attr(study, name, value)

        save("baseline_env_steps", _mean_trial_attr(study, "env_steps"))
        save("baseline_processed_samples", _mean_trial_attr(study, "processed_samples"))
        save("robust_best_params", {})
        save("robust_best_objective_score", _mean_trial_value(study))
        save("robust_best_gym_score", _mean_trial_attr(study, "gym_score"))
        save("robust_best_training_effort", 1.0)
        if sync_fn is not None:
            sync_fn()
    return study


def neighbors(value: Any, choices: Sequence[Any]) -> list[Any]:
    """Return value plus its direct neighbors in choices."""
    index = choices.index(value)
    start = max(0, index - 1)
    end = min(len(choices), index + 2)
    return list(choices[start:end])


def select_robust_best(
    *,
    study: Any,
    search_space: Any,
    incumbent_params: dict[str, Any],
    environment_factory: EnvironmentFactory,
    trial_cfg: TrialConfig,
    scoring_cfg: ScoringConfig,
    base_seed: int = 42,
    top_n: int = 3,
    extra_seeds: Iterable[int] = (1001, 1002),
    progress_fn: ProgressFn | None = None,
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
    best_gym_mean = None
    best_effort_mean = None
    candidate_seed_scores = [
        [float(trial.value)]
        for trial in candidates
    ]

    def score_candidate(
        trial: Any,
        candidate_index: int,
    ) -> tuple[dict[str, Any], float, float, float]:
        scores = [float(trial.value)]
        gym_scores = [float(trial.user_attrs["gym_score"])]
        efforts = [float(trial.user_attrs["training_effort"])]

        for seed_index, seed_offset in enumerate(extra_seeds, start=1):
            if progress_fn is not None:
                progress_fn(
                    candidate_index=candidate_index,
                    candidate_count=len(candidates),
                    seed_index=seed_index,
                    seed_count=len(extra_seeds),
                    candidate_seed_scores=candidate_seed_scores,
                )
            objective = create_objective(
                search_space=search_space,
                incumbent_params=incumbent_params,
                environment_factory=environment_factory,
                trial_cfg=TrialConfig(
                    num_envs=trial_cfg.num_envs,
                    seed=base_seed + seed_offset,
                    device=trial_cfg.device,
                ),
                scoring_cfg=scoring_cfg,
            )
            fixed_trial = _FixedParamTrial(trial.params)
            scores.append(objective(fixed_trial))
            candidate_seed_scores[candidate_index - 1] = list(scores)
            gym_scores.append(float(fixed_trial.user_attrs["gym_score"]))
            efforts.append(float(fixed_trial.user_attrs["training_effort"]))
            if progress_fn is not None:
                progress_fn(
                    candidate_index=candidate_index,
                    candidate_count=len(candidates),
                    seed_index=seed_index,
                    seed_count=len(extra_seeds),
                    candidate_seed_scores=candidate_seed_scores,
                )

        mean_score = sum(scores) / len(scores)
        return (
            dict(trial.params),
            mean_score,
            sum(gym_scores) / len(gym_scores),
            sum(efforts) / len(efforts),
        )

    for candidate_index, trial in enumerate(candidates, start=1):
        params, mean_score, mean_gym_score, mean_effort = score_candidate(
            trial,
            candidate_index,
        )
        if mean_score > best_mean:
            best_mean = mean_score
            best_params = params
            best_gym_mean = mean_gym_score
            best_effort_mean = mean_effort

    selected_params = best_params or {}
    _set_study_user_attr(study, "robust_best_params", selected_params)
    _set_study_user_attr(study, "robust_best_objective_score", best_mean)
    _set_study_user_attr(study, "robust_best_gym_score", best_gym_mean)
    _set_study_user_attr(study, "robust_best_training_effort", best_effort_mean)
    return selected_params


def _top_complete_trials(study: Any, top_n: int) -> list[Any]:
    trials = [
        trial for trial in study.trials
        if _trial_state_name(trial) == "COMPLETE" and trial.value is not None
    ]
    trials.sort(key=lambda trial: trial.value, reverse=True)
    return trials[:top_n]


@log_call
def _create_study(**kwargs) -> Any:
    import optuna

    return optuna.create_study(**kwargs)


def _set_study_user_attr(study: Any, name: str, value: Any) -> None:
    if hasattr(study, "set_user_attr"):
        study.set_user_attr(name, value)
    else:
        study.user_attrs = getattr(study, "user_attrs", {})
        study.user_attrs[name] = value


def _set_or_check_study_attrs(study: Any, attrs: dict[str, Any]) -> None:
    for name, value in attrs.items():
        if name in study.user_attrs and study.user_attrs[name] != value:
            raise ValueError(f"study {name} does not match current configuration")
        _set_study_user_attr(study, name, value)


def _mean_trial_attr(study: Any, name: str) -> float:
    values = [
        float(trial.user_attrs[name])
        for trial in study.trials
        if _trial_state_name(trial) == "COMPLETE" and name in trial.user_attrs
    ]
    if not values:
        raise ValueError(f"study has no complete trials with {name}")
    return sum(values) / len(values)


def _mean_trial_value(study: Any) -> float:
    values = [
        float(trial.value)
        for trial in study.trials
        if _trial_state_name(trial) == "COMPLETE" and trial.value is not None
    ]
    if not values:
        raise ValueError("study has no complete trial values")
    return sum(values) / len(values)


def _trial_state_name(trial: Any) -> str:
    state = trial.state
    return state.name if hasattr(state, "name") else str(state)


class _FixedParamTrial:
    number = 0

    def __init__(self, params: dict[str, Any]) -> None:
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
