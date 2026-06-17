"""Notebook reporting helpers for HPO studies."""

from typing import Any


def plot_lander_progress(study: Any) -> Any:
    """Plot one Lander History point per study."""
    import matplotlib.pyplot as plt

    mean_training_minutes = []
    eval_scores = []
    labels = []

    for index, current_study in enumerate(_study_list(study)):
        point = _study_progress_point(current_study)
        if point is None:
            continue
        mean_training_minutes.append(point["mean_wall_time_seconds"] / 60)
        eval_scores.append(point["eval_score"])
        labels.append(_study_label(current_study, index))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(mean_training_minutes, eval_scores, marker="o", label="Greedy eval score")
    for x, y, label in zip(mean_training_minutes, eval_scores, labels, strict=True):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.axhline(200, color="gray", linestyle="--", label="200")
    ax.axhline(250, color="red", linestyle="--", label="250")
    ax.set_title("Lander History")
    ax.set_xlabel("Mean L4 training time per Lander (min)")
    ax.set_ylabel("Greedy eval score")
    ax.legend()
    fig.tight_layout()
    return fig


def show_lander_live_progress(
    study: Any,
    *,
    target_trials: int,
    lander_studies: Any,
) -> None:
    """Display live Lander History and current Optuna History."""
    _clear_output(wait=True)
    print(f"Target trials: {target_trials}")
    print(f"Finished trials: {finished_trial_count(study)}")
    print("LH: Lander History")
    _display(plot_lander_progress(lander_studies))
    print("OH: Optuna History")
    _display(_plot_optimization_history(study))


def finished_trial_count(study: Any) -> int:
    """Return the number of complete or pruned trials in a study."""
    return _trial_count(study, "COMPLETE") + _trial_count(study, "PRUNED")


def show_study_progress(
    study: Any,
    *,
    target_trials: int,
) -> None:
    """Display current optimization progress in a notebook."""
    _clear_output(wait=True)

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
    best_window = best_trial.user_attrs["best_window"]
    print(f"Best mean return: {best_trial.value:.1f}")
    print(
        "Best episode window:",
        best_window["start_episode"],
        "-",
        best_window["end_episode"],
    )
    print("Best params:")
    _display(best_trial.params)

    fig = _plot_optimization_history(study)
    fig.update_layout(width=1000, height=450, margin=dict(r=180))
    fig.update_xaxes(range=[0, target_trials])
    _display(fig)


def _clear_output(*args, **kwargs) -> None:
    from IPython.display import clear_output

    clear_output(*args, **kwargs)


def _display(value: Any) -> None:
    from IPython.display import display

    display(value)


def _plot_optimization_history(study: Any) -> Any:
    from optuna.visualization import plot_optimization_history

    return plot_optimization_history(study)


def _trial_count(study: Any, state_name: str) -> int:
    return sum(1 for trial in study.trials if _trial_state_name(trial) == state_name)


def _study_list(studies: Any) -> list[Any]:
    if hasattr(studies, "trials"):
        return [studies]
    return list(studies)


def _study_progress_point(study: Any) -> dict[str, float] | None:
    trials = [
        trial for trial in sorted(study.trials, key=lambda trial: trial.number)
        if _trial_state_name(trial) == "COMPLETE"
        and "wall_time_seconds" in trial.user_attrs
    ]
    if not trials:
        return None

    eval_score = _robust_eval_score(study)
    if eval_score is None:
        eval_trials = [
            trial for trial in trials
            if trial.value is not None and "eval_score" in trial.user_attrs
        ]
        if not eval_trials:
            return None
        best_trial = max(eval_trials, key=lambda trial: trial.value)
        eval_score = float(best_trial.user_attrs["eval_score"])

    return {
        "mean_wall_time_seconds": sum(
            float(trial.user_attrs["wall_time_seconds"]) for trial in trials
        ) / len(trials),
        "eval_score": eval_score,
    }


def _robust_eval_score(study: Any) -> float | None:
    user_attrs = getattr(study, "user_attrs", {})
    if "robust_best_eval_score" not in user_attrs:
        return None
    return float(user_attrs["robust_best_eval_score"])


def _study_label(study: Any, index: int) -> str:
    name = getattr(study, "study_name", "")
    if name.startswith("s") and "_" in name:
        return name.split("_", 1)[0].upper()
    return name or f"S{index}"


def _trial_state_name(trial: Any) -> str:
    state = trial.state
    return state.name if hasattr(state, "name") else str(state)
