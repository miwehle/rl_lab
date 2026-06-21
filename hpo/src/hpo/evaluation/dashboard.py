"""Tell the story of an HPO study series in one notebook dashboard.

Study Series shows the overall progress, centered on the Best HPs beside it.
Study follows the current optimization, and HP Robustness Evaluation confirms
the best candidates at the end of each study.
"""

from typing import Any


def build_dashboard(
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
    """Build the study-series dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"secondary_y": True}, {"type": "domain"}],
            [{"secondary_y": True}, {}],
        ],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Study Series",
            "Best HPs",
            f"Study: {_study_title(study)}",
            "HP Robustness Evaluation",
        ),
    )
    _add_study_series(figure, lander_studies)
    _add_best_hps(figure, incumbent_params, study)
    if candidate_seed_scores is None:
        _add_current_study(figure, study, target_trials)
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
        _add_robustness_evaluation(
            figure,
            candidate_seed_scores=candidate_seed_scores,
            candidate_index=candidate_index,
        )
        figure.layout.annotations[3].text = (
            "HP Robustness Evaluation · "
            f"Candidate {candidate_index}/{candidate_count} · "
            f"Seed {seed_index}/{seed_count}"
        )

    _style_dashboard(figure)
    return figure


def _style_dashboard(figure: Any) -> None:
    figure.update_layout(
        width=1200,
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


def _study_title(study: Any) -> str:
    parts = getattr(study, "study_name", "").split("_")
    if not parts or not parts[0]:
        return "Unnamed"
    if len(parts) > 1 and parts[1] == "qe":
        del parts[1]
    return " ".join([parts[0].upper(), *parts[1:]]).replace("_", " ").title()


def show_dashboard_during_optimization(
    study: Any,
    *,
    target_trials: int,
    lander_studies: Any,
    incumbent_params: dict[str, Any],
) -> None:
    """Update the fixed dashboard during Optuna optimization."""
    _clear_output(wait=True)
    _display(
        build_dashboard(
            study=study,
            target_trials=target_trials,
            lander_studies=lander_studies,
            incumbent_params=incumbent_params,
        )
    )


def show_dashboard_during_robustness_evaluation(
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
        build_dashboard(
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


def _clear_output(*args, **kwargs) -> None:
    from IPython.display import clear_output

    clear_output(*args, **kwargs)


def _display(value: Any) -> None:
    from IPython.display import display

    display(value)


def _add_study_series(figure: Any, studies: Any) -> None:
    import plotly.graph_objects as go

    points = _study_series_points(studies)
    efforts = [point["training_effort"] for point in points]
    gym_scores = [point["gym_score"] for point in points]
    qe_scores = [point["qe_score"] for point in points]
    labels = [point["label"] for point in points]
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
                showlegend=False,
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

def _study_series_points(studies: Any) -> list[dict[str, Any]]:
    points = []
    for index, study in enumerate(studies):
        user_attrs = getattr(study, "user_attrs", {})
        gym_score = user_attrs.get("robust_best_gym_score")
        qe_score = user_attrs.get("robust_best_objective_score")
        training_effort = user_attrs.get("robust_best_training_effort")
        if gym_score is None or qe_score is None or training_effort is None:
            continue

        name = getattr(study, "study_name", "")
        label = (
            name.split("_", 1)[0].upper()
            if name.startswith("s") and "_" in name
            else name or f"S{index}"
        )
        points.append({
            "training_effort": float(training_effort),
            "gym_score": float(gym_score),
            "qe_score": float(qe_score),
            "label": label,
        })
    return points


def _add_best_hps(
    figure: Any,
    incumbent_params: dict[str, Any],
    study: Any,
) -> None:
    import plotly.graph_objects as go

    names = list(incumbent_params)
    optimized_params = {
        name
        for trial in study.trials
        for name in trial.params
    }
    row_colors = [
        "#fff2cc" if name in optimized_params else "white"
        for name in names
    ]
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
                    [str(incumbent_params[name]) for name in names],
                ],
                align=["left", "right"],
                fill_color=[row_colors, row_colors],
                height=22,
            ),
        ),
        row=1,
        col=2,
    )


def _add_current_study(
    figure: Any,
    study: Any,
    target_trials: int,
) -> None:
    import plotly.graph_objects as go

    trials = [
        trial for trial in study.trials
        if trial.state.name == "COMPLETE"
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


def _add_robustness_evaluation(
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
            zip(scores, _robustness_offsets(len(scores)), strict=True)
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
        title_text="Candidate",
        tickmode="array",
        tickvals=list(range(1, len(means) + 1)),
        ticktext=list(range(1, len(means) + 1)),
        range=[0.5, len(means) + 0.5],
        row=2,
        col=2,
    )
    figure.update_yaxes(title_text="QE score", row=2, col=2)

def _robustness_offsets(count: int) -> list[float]:
    if count == 1:
        return [0.0]
    center = (count - 1) / 2
    return [(index - center) * 0.06 for index in range(count)]
