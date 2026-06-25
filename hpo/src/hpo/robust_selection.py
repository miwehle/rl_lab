"""Robust selection of HPO study candidates."""

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

from hpo.objective import ObjectiveConfig, create_objective
from hpo.study_reporting import RobustnessProgress


RobustnessProgressFn = Callable[[RobustnessProgress], None]


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
    robustness_checkpoints = []
    candidate_seed_scores = [
        [float(trial.value)]
        for trial in candidates
    ]

    def save(key, value):
        study.set_user_attr(key, value)

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
            score = objective(fixed_trial)
            scores.append(score)
            robustness_checkpoints.append(
                {
                    "trial_number": trial.number,
                    "seed_offset": seed_offset,
                    "score": score,
                } | fixed_trial.user_attrs
            )
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
    save("robust_best_params", selected_params)
    save("robust_best_score", best_mean)
    save("robustness_checkpoints", robustness_checkpoints)
    return selected_params


def _top_complete_trials(study: Any, top_n: int) -> list[Any]:
    trials = [
        trial for trial in study.trials
        if trial.state.name == "COMPLETE" and trial.value is not None
    ]
    trials.sort(key=lambda trial: trial.value, reverse=True)
    return trials[:top_n]


class _FixedParamTrial:
    """Double for an Optuna Trial during robustness evaluation.

    It provides a fixed HP set through the suggest_* methods, but gets its own
    trial number for checkpoint files, so robustness evaluation can use the same
    objective and checkpointing chain as normal training trials.
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
