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
    score_window: int,
    study_dir: str | Path,
    device,
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
        score_window=score_window,
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

    while finished_trial_count(study) < n_trials:
        logger.info("study.optimize")
        study.optimize(objective, n_trials=1)
        if progress_fn is not None:
            progress_fn(study, target_trials=n_trials)

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
    score_window: int,
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
    best_eval_mean = None

    def score_candidate(trial: Any) -> tuple[dict[str, Any], float, float | None]:
        scores = [float(trial.value)]
        eval_scores = []
        if "eval_score" in getattr(trial, "user_attrs", {}):
            eval_scores.append(float(trial.user_attrs["eval_score"]))

        for seed_offset in extra_seeds:
            objective = create_objective(
                search_space=search_space_factory(),
                num_episodes=num_episodes,
                score_window=score_window,
                seed=base_seed + seed_offset,
                device=device,
                num_envs=num_envs,
            )
            fixed_trial = _FixedParamTrial(trial.params)
            scores.append(objective(fixed_trial))
            if "eval_score" in fixed_trial.user_attrs:
                eval_scores.append(float(fixed_trial.user_attrs["eval_score"]))

        mean_score = sum(scores) / len(scores)
        mean_eval_score = (
            sum(eval_scores) / len(eval_scores)
            if eval_scores else None
        )
        return dict(trial.params), mean_score, mean_eval_score

    for trial in candidates:
        params, mean_score, mean_eval_score = score_candidate(trial)
        if mean_score > best_mean:
            best_mean = mean_score
            best_params = params
            best_eval_mean = mean_eval_score

    selected_params = best_params or {}
    _set_study_user_attr(study, "robust_best_params", selected_params)
    _set_study_user_attr(study, "robust_best_objective_score", best_mean)
    if best_eval_mean is not None:
        _set_study_user_attr(study, "robust_best_eval_score", best_eval_mean)
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
