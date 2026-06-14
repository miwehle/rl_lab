"""Notebook reporting helpers for HPO studies."""

from collections.abc import Callable
from typing import Any


def finished_trial_count(study: Any) -> int:
    """Return the number of complete or pruned trials in a study."""
    return _trial_count(study, "COMPLETE") + _trial_count(study, "PRUNED")


def show_study_progress(
    study: Any,
    *,
    target_trials: int,
    clear_output_fn: Callable[..., None] | None = None,
    display_fn: Callable[[Any], None] | None = None,
    plot_history: Callable[[Any], Any] | None = None,
) -> None:
    """Display current optimization progress in a notebook."""
    if clear_output_fn is None or display_fn is None:
        from IPython.display import clear_output, display

        clear_output_fn = clear_output if clear_output_fn is None else clear_output_fn
        display_fn = display if display_fn is None else display_fn

    clear_output_fn(wait=True)

    complete_trials = _trial_count(study, "COMPLETE")
    pruned_trials = _trial_count(study, "PRUNED")
    finished_trials = complete_trials + pruned_trials

    print(f"Target trials: {target_trials}")
    print(f"Finished trials: {finished_trials}")
    print(f"Complete trials: {complete_trials}")
    print(f"Pruned trials: {pruned_trials}")
    print(
        "Episodes saved by pruning:",
        sum(trial.user_attrs.get("episodes_saved_by_pruning", 0) for trial in study.trials),
    )

    if complete_trials == 0:
        print("Best mean return: no complete trials yet")
        return

    best_trial = study.best_trial
    print(f"Best mean return: {best_trial.value:.1f}")
    print(
        "Best episode window:",
        best_trial.user_attrs["best_window_start_episode"],
        "-",
        best_trial.user_attrs["best_window_end_episode"],
    )
    print("Best params:")
    display_fn(best_trial.params)

    if plot_history is None:
        from optuna.visualization import plot_optimization_history

        plot_history = plot_optimization_history

    fig = plot_history(study)
    fig.update_layout(width=1000, height=450, margin=dict(r=180))
    fig.update_xaxes(range=[0, target_trials])
    display_fn(fig)


def _trial_count(study: Any, state_name: str) -> int:
    return sum(1 for trial in study.trials if _trial_state_name(trial) == state_name)


def _trial_state_name(trial: Any) -> str:
    state = trial.state
    return state.name if hasattr(state, "name") else str(state)
