"""Notebook reporting helpers for HPO studies."""

from collections.abc import Callable
from typing import Any


def plot_lander_progress(study: Any) -> Any:
    """Plot greedy eval score over cumulative training time."""
    import matplotlib.pyplot as plt

    elapsed_minutes = []
    eval_scores = []
    cumulative_seconds = 0.0

    for trial in _progress_trials(study):
        if _trial_state_name(trial) != "COMPLETE":
            continue
        if "wall_time_seconds" not in trial.user_attrs:
            continue
        if "eval_score" not in trial.user_attrs:
            continue

        cumulative_seconds += float(trial.user_attrs["wall_time_seconds"])
        elapsed_minutes.append(cumulative_seconds / 60)
        eval_scores.append(float(trial.user_attrs["eval_score"]))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(elapsed_minutes, eval_scores, marker="o", label="Greedy eval score")
    ax.axhline(200, color="gray", linestyle="--", label="200")
    ax.axhline(250, color="red", linestyle="--", label="250")
    ax.set_title("LunarLander progress")
    ax.set_xlabel("Cumulative L4 training time (min)")
    ax.set_ylabel("Greedy eval score")
    ax.legend()
    fig.tight_layout()
    return fig


def show_lander_live_progress(
    study: Any,
    *,
    target_trials: int,
    lander_studies: Any,
    clear_output_fn: Callable[..., None] | None = None,
    display_fn: Callable[[Any], None] | None = None,
    plot_history: Callable[[Any], Any] | None = None,
) -> None:
    """Display live Lander History and current Optuna History."""
    if clear_output_fn is None or display_fn is None:
        from IPython.display import clear_output, display

        clear_output_fn = clear_output if clear_output_fn is None else clear_output_fn
        display_fn = display if display_fn is None else display_fn

    if plot_history is None:
        from optuna.visualization import plot_optimization_history

        plot_history = plot_optimization_history

    clear_output_fn(wait=True)
    print(f"Target trials: {target_trials}")
    print(f"Finished trials: {finished_trial_count(study)}")
    print("LH: Lander History")
    display_fn(plot_lander_progress(lander_studies))
    print("OH: Optuna History")
    display_fn(plot_history(study))


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


def _progress_trials(studies: Any) -> list[Any]:
    if hasattr(studies, "trials"):
        studies = [studies]

    trials = []
    for study in studies:
        trials.extend(sorted(study.trials, key=lambda trial: trial.number))
    return trials


def _trial_state_name(trial: Any) -> str:
    state = trial.state
    return state.name if hasattr(state, "name") else str(state)
