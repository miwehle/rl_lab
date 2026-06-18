"""Study orchestration helpers for LunarLander HPO."""

import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from hpo.evaluation.reporting import finished_trial_count, show_study_progress
from hpo.lunar_lander.logging import log_call
from hpo.lunar_lander.objective import create_objective


logger = logging.getLogger(__name__)

SearchSpaceFactory = Callable[[], Any]
ProgressFn = Callable[..., None]


@log_call
def run_study(
    *,
    study_name: str,
    search_space: Any,
    n_trials: int,
    num_episodes: int,
    study_dir: str | Path,
    device,
    baseline_env_steps: float | None = None,
    baseline_processed_samples: float | None = None,
    alpha: float = 0.5,
    quality_weight: float = 0.9,
    quality_min: float = 200.0,
    quality_target: float = 250.0,
    eval_episodes: int = 20,
    eval_seed: int = 10_000,
    num_envs: int = 16,
    seed: int | None = 42,
    progress_fn: ProgressFn | None = show_study_progress,
) -> Any:
    """Create or load an Optuna study and run it to the target trial count."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    study_path = Path(study_dir)
    study_path.mkdir(parents=True, exist_ok=True)

    objective = create_objective(
        search_space=search_space,
        num_episodes=num_episodes,
        baseline_env_steps=baseline_env_steps,
        baseline_processed_samples=baseline_processed_samples,
        alpha=alpha,
        quality_weight=quality_weight,
        quality_min=quality_min,
        quality_target=quality_target,
        eval_episodes=eval_episodes,
        eval_seed=eval_seed,
        seed=seed,
        device=device,
        num_envs=num_envs,
    )
    study = _create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{study_path / f'{study_name}.db'}",
        load_if_exists=True,
    )
    scoring_attrs = {
        "alpha": alpha,
        "quality_weight": quality_weight,
        "quality_min": quality_min,
        "quality_target": quality_target,
        "eval_episodes": eval_episodes,
        "eval_seeds": list(range(eval_seed, eval_seed + eval_episodes)),
    }
    if baseline_env_steps is not None:
        scoring_attrs["baseline_env_steps"] = baseline_env_steps
        scoring_attrs["baseline_processed_samples"] = baseline_processed_samples
    _set_or_check_study_attrs(study, scoring_attrs)

    while finished_trial_count(study) < n_trials:
        logger.info("study.optimize")
        study.optimize(objective, n_trials=1)
        if progress_fn is not None:
            progress_fn(study, target_trials=n_trials)

    if baseline_env_steps is None:
        _set_study_user_attr(
            study,
            "baseline_env_steps",
            _mean_trial_attr(study, "env_steps"),
        )
        _set_study_user_attr(
            study,
            "baseline_processed_samples",
            _mean_trial_attr(study, "processed_samples"),
        )
        _set_study_user_attr(study, "robust_best_params", {})
        _set_study_user_attr(
            study,
            "robust_best_objective_score",
            _mean_trial_value(study),
        )
        _set_study_user_attr(
            study,
            "robust_best_gym_score",
            _mean_trial_attr(study, "gym_score"),
        )
        _set_study_user_attr(study, "robust_best_training_effort", 1.0)
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
    search_space_factory: SearchSpaceFactory,
    num_episodes: int,
    device,
    num_envs: int = 16,
    base_seed: int = 42,
    top_n: int = 3,
    extra_seeds: Iterable[int] = (1001, 1002),
) -> dict[str, Any]:
    """Re-check top candidates with extra seeds and return the best params."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    candidates = _top_complete_trials(study, top_n)
    if not candidates:
        raise ValueError("study has no complete trials")

    best_params = None
    best_mean = float("-inf")
    best_gym_mean = None
    best_effort_mean = None

    def score_candidate(trial: Any) -> tuple[dict[str, Any], float, float, float]:
        scores = [float(trial.value)]
        gym_scores = [float(trial.user_attrs["gym_score"])]
        efforts = [float(trial.user_attrs["training_effort"])]

        for seed_offset in extra_seeds:
            objective = create_objective(
                search_space=search_space_factory(),
                num_episodes=num_episodes,
                baseline_env_steps=study.user_attrs["baseline_env_steps"],
                baseline_processed_samples=study.user_attrs[
                    "baseline_processed_samples"
                ],
                alpha=study.user_attrs["alpha"],
                quality_weight=study.user_attrs["quality_weight"],
                quality_min=study.user_attrs["quality_min"],
                quality_target=study.user_attrs["quality_target"],
                eval_episodes=study.user_attrs["eval_episodes"],
                eval_seed=study.user_attrs["eval_seeds"][0],
                seed=base_seed + seed_offset,
                device=device,
                num_envs=num_envs,
            )
            fixed_trial = _FixedParamTrial(trial.params)
            scores.append(objective(fixed_trial))
            gym_scores.append(float(fixed_trial.user_attrs["gym_score"]))
            efforts.append(float(fixed_trial.user_attrs["training_effort"]))

        mean_score = sum(scores) / len(scores)
        return (
            dict(trial.params),
            mean_score,
            sum(gym_scores) / len(gym_scores),
            sum(efforts) / len(efforts),
        )

    for trial in candidates:
        params, mean_score, mean_gym_score, mean_effort = score_candidate(trial)
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
