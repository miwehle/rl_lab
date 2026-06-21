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

    fig, gym_ax = plt.subplots(figsize=(11, 2.7))
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
        loc="lower left",
        ncol=4,
        fontsize="small",
    )
    fig.subplots_adjust(left=0.07, right=0.93, top=0.82, bottom=0.22)
    return fig


def show_lander_live_progress(
    study: Any,
    *,
    target_trials: int,
    lander_studies: Any,
    incumbent_params: dict[str, Any],
) -> None:
    """Update the fixed dashboard during Optuna optimization."""
    _clear_output(wait=True)
    _display(
        _dashboard_figure(
            study=study,
            target_trials=target_trials,
            lander_studies=lander_studies,
            incumbent_params=incumbent_params,
        )
    )


def show_robustness_progress(
    study: Any,
    *,
    lander_studies: Any,
    incumbent_params: dict[str, Any],
    candidate_index: int,
    candidate_count: int,
    seed_index: int,
    seed_count: int,
    candidate_seed_scores: list[list[float]],
) -> None:
    """Update the fixed dashboard during robustness evaluation."""
    _clear_output(wait=True)
    _display(
        _dashboard_figure(
            study=study,
            target_trials=len(study.trials),
            lander_studies=lander_studies,
            incumbent_params=incumbent_params,
            candidate_index=candidate_index,
            candidate_count=candidate_count,
            seed_index=seed_index,
            seed_count=seed_count,
            candidate_seed_scores=candidate_seed_scores,
        )
    )


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


def _dashboard_figure(
    *,
    study: Any,
    target_trials: int,
    lander_studies: Any,
    incumbent_params: dict[str, Any],
    candidate_index: int | None = None,
    candidate_count: int | None = None,
    seed_index: int | None = None,
    seed_count: int | None = None,
    candidate_seed_scores: list[list[float]] | None = None,
) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"secondary_y": True}, {"type": "domain"}],
            [{"secondary_y": True}, {}],
        ],
        row_heights=[0.42, 0.58],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Study Series",
            "Best HPs (Current Incumbent)",
            f"Study: {_study_title(study)}",
            "HP Robustness Evaluation",
        ),
    )
    _add_lander_history(figure, lander_studies)
    _add_incumbent_table(figure, incumbent_params)
    if candidate_seed_scores is None:
        _add_optimization_history(figure, study, target_trials)
        figure.add_annotation(
            text="Waiting for robustness evaluation",
            row=2,
            col=2,
            showarrow=False,
            font=dict(color="gray"),
        )
    else:
        figure.add_annotation(
            text="Waiting for optimization",
            row=2,
            col=1,
            showarrow=False,
            font=dict(color="gray"),
        )
        _add_podium(
            figure,
            candidate_seed_scores=candidate_seed_scores,
            candidate_index=candidate_index,
        )
        figure.layout.annotations[3].text = (
            "HP Robustness Evaluation · "
            f"Candidate {candidate_index}/{candidate_count} · "
            f"Seed {seed_index}/{seed_count}"
        )

    figure.update_layout(
        width=1100,
        height=650,
        margin=dict(l=70, r=70, t=55, b=55),
        legend=dict(
            orientation="h",
            x=0.01,
            y=0.61,
            xanchor="left",
            yanchor="bottom",
        ),
        plot_bgcolor="white",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    figure.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return figure


def _add_incumbent_table(
    figure: Any,
    incumbent_params: dict[str, Any],
) -> None:
    import plotly.graph_objects as go

    names = list(incumbent_params)
    figure.add_trace(
        go.Table(
            columnwidth=[1.5, 1.0],
            header=dict(
                values=["HP", "Value"],
                align=["left", "right"],
                fill_color="#e8eef7",
            ),
            cells=dict(
                values=[
                    names,
                    [_format_hp_value(incumbent_params[name]) for name in names],
                ],
                align=["left", "right"],
                fill_color="white",
                height=22,
            ),
        ),
        row=1,
        col=2,
    )


def _format_hp_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return f"{value:_}"
    return str(value)


def _add_lander_history(figure: Any, studies: Any) -> None:
    import plotly.graph_objects as go

    points = []
    for index, study in enumerate(_study_list(studies)):
        point = _study_progress_point(study)
        if point is not None:
            points.append((point, _study_label(study, index)))

    efforts = [point["training_effort"] for point, _ in points]
    gym_scores = [point["gym_score"] for point, _ in points]
    qe_scores = [point["qe_score"] for point, _ in points]
    labels = [label for _, label in points]
    if efforts:
        effort_margin = max(0.05, 0.05 * (max(efforts) - min(efforts)))
        effort_range = [
            min(efforts) - effort_margin,
            max(efforts) + effort_margin,
        ]
    else:
        effort_range = [0.95, 1.05]
    figure.add_trace(
        go.Scatter(
            x=efforts,
            y=gym_scores,
            mode="lines+markers+text",
            text=labels,
            textposition="top right",
            name="Gym score",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=efforts,
            y=qe_scores,
            mode="lines+markers",
            name="QE score",
            line=dict(color="#ff7f0e"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    for value, name, dash in ((200, "Gym 200", "dash"), (250, "Gym 250", "dot")):
        figure.add_trace(
            go.Scatter(
                x=effort_range,
                y=[value, value],
                mode="lines",
                name=name,
                line=dict(color="gray", dash=dash),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    figure.update_xaxes(
        title_text="Training effort relative to S0",
        range=effort_range,
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="Gym score", row=1, col=1, secondary_y=False)
    figure.update_yaxes(title_text="QE score", row=1, col=1, secondary_y=True)


def _add_optimization_history(
    figure: Any,
    study: Any,
    target_trials: int,
) -> None:
    import plotly.graph_objects as go

    trials = [
        trial for trial in study.trials
        if _trial_state_name(trial) == "COMPLETE"
        and trial.value is not None
        and "gym_score" in trial.user_attrs
    ]
    hover_params = [
        "".join(f"<br>{name}: {value}" for name, value in trial.params.items())
        for trial in trials
    ]
    numbers = [trial.number for trial in trials]
    qe_scores = [float(trial.value) for trial in trials]
    best_scores = []
    best = float("-inf")
    for score in qe_scores:
        best = max(best, score)
        best_scores.append(best)

    figure.add_trace(
        go.Scatter(
            x=numbers,
            y=[trial.user_attrs["gym_score"] for trial in trials],
            mode="markers",
            name="Gym score",
            showlegend=False,
            marker=dict(color="#1f77b4"),
            customdata=hover_params,
            hovertemplate=(
                "Trial: %{x}<br>Gym score: %{y:.1f}"
                "%{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=numbers,
            y=qe_scores,
            mode="markers",
            name="QE score",
            showlegend=False,
            marker=dict(color="#ff7f0e"),
            customdata=hover_params,
            hovertemplate=(
                "Trial: %{x}<br>QE score: %{y:.3f}"
                "%{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=numbers,
            y=best_scores,
            mode="lines",
            name="Best QE score",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    figure.update_xaxes(
        title_text="Trial",
        range=[0, max(1, target_trials)],
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Gym score", row=2, col=1, secondary_y=False)
    figure.update_yaxes(title_text="QE score", row=2, col=1, secondary_y=True)


def _add_podium(
    figure: Any,
    *,
    candidate_seed_scores: list[list[float]],
    candidate_index: int | None,
) -> None:
    import plotly.graph_objects as go

    point_x = []
    point_y = []
    point_labels = []
    point_colors = []
    for candidate, scores in enumerate(candidate_seed_scores, start=1):
        for seed, (score, offset) in enumerate(
            zip(scores, _centered_offsets(len(scores)), strict=True)
        ):
            point_x.append(candidate + offset)
            point_y.append(score)
            point_labels.append(
                "Optimize trial" if seed == 0 else f"Extra seed {seed}"
            )
            point_colors.append(
                "#1f77b4" if candidate < candidate_index else
                "#ff7f0e" if candidate == candidate_index else
                "lightgray"
            )

    figure.add_trace(
        go.Scatter(
            x=point_x,
            y=point_y,
            mode="markers",
            marker=dict(color=point_colors, size=8),
            customdata=point_labels,
            hovertemplate="%{customdata}<br>QE score: %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    means = [sum(scores) / len(scores) for scores in candidate_seed_scores]
    figure.add_trace(
        go.Scatter(
            x=list(range(1, len(means) + 1)),
            y=means,
            mode="markers",
            marker=dict(color="red", symbol="diamond", size=11),
            hovertemplate="Mean QE score: %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, len(means) + 1)),
        ticktext=[
            f"Candidate {index}" for index in range(1, len(means) + 1)
        ],
        range=[0.5, len(means) + 0.5],
        row=2,
        col=2,
    )
    figure.update_yaxes(title_text="QE score", row=2, col=2)


def _centered_offsets(count: int) -> list[float]:
    if count == 1:
        return [0.0]
    center = (count - 1) / 2
    return [(index - center) * 0.06 for index in range(count)]

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
        width=535,
        height=315,
        margin=dict(l=60, r=65, t=35, b=50),
        showlegend=False,
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
