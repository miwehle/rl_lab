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
    import matplotlib.pyplot as plt

    _clear_output(wait=True)
    print(f"Target trials: {target_trials}")
    print(f"Finished trials: {finished_trial_count(study)}")
    print("LH: Lander History")
    figure = plot_lander_progress(lander_studies)
    _display(figure)
    plt.close(figure)
    for completed_study in reversed(_study_list(lander_studies)):
        if "robust_best_params" in completed_study.user_attrs:
            print("Best hyperparameters:")
            _display(completed_study.user_attrs["robust_best_params"])
            break
    print("OH: Optuna History")
    _display(_optimization_history_figure(study, target_trials))


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
    print(f"Best mean return: {best_trial.value:.1f}")
    print(
        "Best episode window:",
        best_trial.user_attrs["best_window_start_episode"],
        "-",
        best_trial.user_attrs["best_window_end_episode"],
    )
    print("Best params:")
    _display(best_trial.params)

    _display(_optimization_history_figure(study, target_trials))


def _clear_output(*args, **kwargs) -> None:
    from IPython.display import clear_output

    clear_output(*args, **kwargs)


def _display(value: Any) -> None:
    from IPython.display import display

    display(value)


def _plot_optimization_history(study: Any) -> Any:
    from optuna.visualization import plot_optimization_history

    return plot_optimization_history(study)


def _optimization_history_figure(study: Any, target_trials: int) -> Any:
    fig = _plot_optimization_history(study)
    fig.update_layout(width=850, height=430, margin=dict(l=55, r=170, t=70, b=55))
    fig.update_xaxes(range=[0, target_trials], autorange=False)
    return fig


def _trial_count(study: Any, state_name: str) -> int:
    return sum(1 for trial in study.trials if _trial_state_name(trial) == state_name)


def _study_list(studies: Any) -> list[Any]:
    if hasattr(studies, "trials"):
        return [studies]
    return list(studies)


def _study_progress_point(study: Any) -> dict[str, float] | None:
    eval_score = getattr(study, "user_attrs", {}).get("robust_best_eval_score")
    if eval_score is None:
        return None

    trials = [
        trial for trial in study.trials
        if _trial_state_name(trial) == "COMPLETE"
        and "wall_time_seconds" in trial.user_attrs
    ]
    if not trials:
        return None

    return {
        "mean_wall_time_seconds": sum(
            float(trial.user_attrs["wall_time_seconds"]) for trial in trials
        ) / len(trials),
        "eval_score": float(eval_score),
    }


def _study_label(study: Any, index: int) -> str:
    name = getattr(study, "study_name", "")
    if name.startswith("s") and "_" in name:
        return name.split("_", 1)[0].upper()
    return name or f"S{index}"


def _trial_state_name(trial: Any) -> str:
    state = trial.state
    return state.name if hasattr(state, "name") else str(state)
