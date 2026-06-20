"""Notebook reporting helpers for HPO studies."""

from typing import Any


def plot_lander_progress(study: Any) -> Any:
    """Plot Gym and quality-effort progress with one point per study."""
    import matplotlib.pyplot as plt

    training_efforts = []
    gym_scores = []
    qe_scores = []
    labels = []

    for index, current_study in enumerate(_study_list(study)):
        point = _study_progress_point(current_study)
        if point is None:
            continue
        training_efforts.append(point["training_effort"])
        gym_scores.append(point["gym_score"])
        qe_scores.append(point["qe_score"])
        labels.append(_study_label(current_study, index))

    fig, gym_ax = plt.subplots(figsize=(11, 4.3))
    qe_ax = gym_ax.twinx()
    gym_line, = gym_ax.plot(
        training_efforts,
        gym_scores,
        color="tab:blue",
        marker="o",
        label="Gym score",
    )
    qe_line, = qe_ax.plot(
        training_efforts,
        qe_scores,
        color="tab:orange",
        marker="o",
        label="QE score",
    )
    for x, y, label in zip(training_efforts, gym_scores, labels, strict=True):
        gym_ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    gym_200 = gym_ax.axhline(
        200,
        color="gray",
        linestyle="--",
        label="Gym 200",
    )
    gym_250 = gym_ax.axhline(
        250,
        color="gray",
        linestyle=":",
        label="Gym 250",
    )
    gym_ax.set_title("Lander History")
    gym_ax.set_xlabel("Training effort relative to S0")
    gym_ax.set_ylabel("Gym score")
    qe_ax.set_ylabel("QE score")
    gym_ax.legend(
        handles=[gym_line, qe_line, gym_200, gym_250],
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.07, right=0.78, top=0.84, bottom=0.16)
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
    figure = plot_lander_progress(lander_studies)
    _display(figure)
    plt.close(figure)
    for completed_study in reversed(_study_list(lander_studies)):
        if "robust_best_params" in completed_study.user_attrs:
            print("Best hyperparameters:")
            _display(completed_study.user_attrs["robust_best_params"])
            break
    print(f"Study: {_study_title(study)}")
    _display(_optimization_history_figure(study, target_trials))


def plot_robustness_progress(
    candidate_scores: list[float],
    candidate_index: int,
) -> Any:
    """Plot running mean QE scores for robustness candidates."""
    import matplotlib.pyplot as plt

    colors = [
        "tab:blue" if index < candidate_index else
        "tab:orange" if index == candidate_index else
        "lightgray"
        for index in range(1, len(candidate_scores) + 1)
    ]
    score_range = max(candidate_scores) - min(candidate_scores)
    baseline = min(candidate_scores) - max(0.1, 0.1 * score_range)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    bars = ax.bar(
        [f"Candidate {index}" for index in range(1, len(candidate_scores) + 1)],
        [score - baseline for score in candidate_scores],
        bottom=baseline,
        color=colors,
    )
    ax.bar_label(
        bars,
        labels=[f"{score:.3f}" for score in candidate_scores],
        padding=3,
    )
    ax.set_title("Robustness Candidates")
    ax.set_ylabel("Mean QE score")
    fig.subplots_adjust(left=0.07, right=0.78, top=0.82, bottom=0.18)
    return fig


def show_robustness_progress(
    study: Any,
    *,
    lander_studies: Any,
    candidate_index: int,
    candidate_count: int,
    seed_index: int,
    seed_count: int,
    candidate_scores: list[float],
) -> None:
    """Display study history and robustness candidate progress."""
    import matplotlib.pyplot as plt

    _clear_output(wait=True)
    history = plot_lander_progress(lander_studies)
    _display(history)
    plt.close(history)
    print(f"Study: {_study_title(study)}")
    print("Phase: Robustness evaluation")
    print(
        f"Candidate {candidate_index}/{candidate_count} · "
        f"Seed {seed_index}/{seed_count}"
    )
    candidates = plot_robustness_progress(candidate_scores, candidate_index)
    _display(candidates)
    plt.close(candidates)

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

    print(f"Study: {_study_title(study)}")

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
    print(f"Best objective score: {best_trial.value:.3f}")
    print(f"Gym score: {best_trial.user_attrs['gym_score']:.1f}")
    print(f"Training effort: {best_trial.user_attrs['training_effort']:.3f}")
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
    import plotly.graph_objects as go

    fig = _plot_optimization_history(study)
    trials = [
        trial for trial in study.trials
        if _trial_state_name(trial) == "COMPLETE"
        and "gym_score" in trial.user_attrs
    ]
    hover_params = [
        "".join(f"<br>{name}: {value}" for name, value in trial.params.items())
        for trial in trials
    ]

    qe_trace, best_qe_trace = fig.data[:2]
    qe_trace.update(
        name="QE score",
        marker_color="#ff7f0e",
        legendrank=2,
        yaxis="y2",
        customdata=hover_params,
        hovertemplate="Trial: %{x}<br>QE score: %{y:.3f}%{customdata}<extra></extra>",
    )
    best_qe_trace.update(
        name="Best QE score",
        line_color="red",
        legendrank=3,
        yaxis="y2",
        hovertemplate="Trial: %{x}<br>Best QE score: %{y:.3f}<extra></extra>",
    )

    fig.add_trace(go.Scatter(
        x=[trial.number for trial in trials],
        y=[trial.user_attrs["gym_score"] for trial in trials],
        mode="markers",
        name="Gym score",
        marker=dict(color="#1f77b4"),
        legendrank=1,
        yaxis="y",
        customdata=hover_params,
        hovertemplate="Trial: %{x}<br>Gym score: %{y:.1f}%{customdata}<extra></extra>",
    ))
    fig.add_hline(y=200, line_color="gray", line_dash="dash")
    fig.add_hline(y=250, line_color="gray", line_dash="dot")
    fig.update_layout(
        width=1100,
        height=430,
        margin=dict(l=70, r=250, t=70, b=55),
        legend=dict(x=1.08, xanchor="left"),
        yaxis=dict(title="Gym score"),
        yaxis2=dict(title="QE score", overlaying="y", side="right"),
    )
    fig.update_xaxes(range=[0, target_trials], autorange=False)
    return fig


def _trial_count(study: Any, state_name: str) -> int:
    return sum(1 for trial in study.trials if _trial_state_name(trial) == state_name)


def _study_list(studies: Any) -> list[Any]:
    if hasattr(studies, "trials"):
        return [studies]
    return list(studies)


def _study_progress_point(study: Any) -> dict[str, float] | None:
    user_attrs = getattr(study, "user_attrs", {})
    gym_score = user_attrs.get("robust_best_gym_score")
    qe_score = user_attrs.get("robust_best_objective_score")
    training_effort = user_attrs.get("robust_best_training_effort")
    if gym_score is None or qe_score is None or training_effort is None:
        return None

    return {
        "training_effort": float(training_effort),
        "gym_score": float(gym_score),
        "qe_score": float(qe_score),
    }


def _study_title(study: Any) -> str:
    parts = getattr(study, "study_name", "").split("_")
    if not parts or not parts[0]:
        return "Unnamed"
    if len(parts) > 1 and parts[1] == "qe":
        del parts[1]
    return " ".join([parts[0].upper(), *parts[1:]]).replace("_", " ").title()

def _study_label(study: Any, index: int) -> str:
    name = getattr(study, "study_name", "")
    if name.startswith("s") and "_" in name:
        return name.split("_", 1)[0].upper()
    return name or f"S{index}"


def _trial_state_name(trial: Any) -> str:
    state = trial.state
    return state.name if hasattr(state, "name") else str(state)
