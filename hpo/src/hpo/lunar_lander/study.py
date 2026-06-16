"""Study orchestration helpers for LunarLander HPO."""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from hpo.evaluation.reporting import finished_trial_count, show_study_progress
from hpo.lunar_lander.objective import create_objective


SearchSpaceFactory = Callable[[], Any]
ProgressFn = Callable[..., None]


def run_study(
    *,
    study_name: str,
    search_space: Any,
    n_trials: int,
    num_episodes: int,
    score_window: int,
    output_dir: str | Path,
    study_dir: str | Path,
    device,
    num_envs: int = 16,
    seed: int | None = 42,
    progress_fn: ProgressFn | None = show_study_progress,
) -> Any:
    """Create or load an Optuna study and run it to the target trial count."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
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

    for trial in candidates:
        scores = [float(trial.value)]
        for seed_offset in extra_seeds:
            objective = create_objective(
                search_space=search_space_factory(),
                num_episodes=num_episodes,
                score_window=score_window,
                seed=base_seed + seed_offset,
                device=device,
                num_envs=num_envs,
            )
            scores.append(objective(_FixedParamTrial(trial.params)))

        mean_score = sum(scores) / len(scores)
        if mean_score > best_mean:
            best_mean = mean_score
            best_params = dict(trial.params)

    return best_params or {}


def _top_complete_trials(study: Any, top_n: int) -> list[Any]:
    trials = [
        trial for trial in study.trials
        if _trial_state_name(trial) == "COMPLETE" and trial.value is not None
    ]
    trials.sort(key=lambda trial: trial.value, reverse=True)
    return trials[:top_n]


def _create_study(**kwargs) -> Any:
    import optuna

    return optuna.create_study(**kwargs)


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
