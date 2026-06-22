"""Tell the story of an HPO study series in one notebook dashboard.

Study Series shows the overall progress, centered on the Best HPs beside it.
Study follows the current optimization, and HP Robustness Evaluation confirms
the best candidates at the end of each study.
"""

from typing import Any

from hpo.study_reporting import RobustnessProgress, StudySeriesReporter


def build_dashboard(
    *,
    study: Any,
    target_trials: int,
    studies: Any,
    incumbent_params: dict[str, Any],
    robustness_progress: RobustnessProgress | None = None,
) -> Any:
    """Build the study-series dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{}, {"type": "domain"}],
            [{}, {}],
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
    _add_study_series(figure, studies)
    _add_best_hps(figure, incumbent_params, study)
    if robustness_progress is None:
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
            candidate_seed_scores=robustness_progress.candidate_seed_scores,
            candidate_index=robustness_progress.candidate_index,
        )
        figure.layout.annotations[3].text = (
            "HP Robustness Evaluation · "
            f"Candidate {robustness_progress.candidate_index}/"
            f"{robustness_progress.candidate_count} · "
            f"Seed {robustness_progress.seed_index}/"
            f"{robustness_progress.seed_count}"
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
    return " ".join([parts[0].upper(), *parts[1:]]).replace("_", " ").title()


class Dashboard(StudySeriesReporter):
    """Report study-series progress through the notebook dashboard."""

    def report_optimization(
        self,
        study: Any,
        *,
        target_trials: int,
        studies: list[Any],
        incumbent_params: dict[str, Any],
    ) -> None:
        _clear_output(wait=True)
        _display(
            build_dashboard(
                study=study,
                target_trials=target_trials,
                studies=studies,
                incumbent_params=incumbent_params,
            )
        )

    def report_robustness_evaluation(
        self,
        study: Any,
        *,
        studies: list[Any],
        incumbent_params: dict[str, Any],
        progress: RobustnessProgress,
    ) -> None:
        _clear_output(wait=True)
        _display(
            build_dashboard(
                study=study,
                target_trials=len(study.trials),
                studies=studies,
                incumbent_params=incumbent_params,
                robustness_progress=progress,
            )
        )


def _clear_output(*args, **kwargs) -> None:
    from IPython.display import clear_output

    clear_output(*args, **kwargs)


def _display(value: Any) -> None:
    from IPython.display import display

    display(value)


# The code tells the same story as the dashboard:
# - Four _add_* functions, one per panel.
# - Presentation and data are kept separate.


def _add_study_series(figure: Any, studies: Any) -> None:
    import plotly.graph_objects as go

    points = _study_series_points(studies)
    scores = [point["score"] for point in points]
    labels = [point["label"] for point in points]
    x = list(range(len(points)))
    figure.add_trace(
        go.Scatter(
            x=x,
            y=scores,
            mode="lines+markers+text",
            text=labels,
            textposition="top right",
            name="Score",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    x_range = [-0.5, max(0.5, len(points) - 0.5)]
    for value, name, dash in ((200, "Score 200", "dash"), (250, "Score 250", "dot")):
        figure.add_trace(
            go.Scatter(
                x=x_range,
                y=[value, value],
                mode="lines",
                name=name,
                showlegend=False,
                line=dict(color="gray", dash=dash),
            ),
            row=1,
            col=1,
        )
    figure.update_xaxes(
        title_text="Study",
        tickmode="array",
        tickvals=x,
        ticktext=labels,
        range=x_range,
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="Score", row=1, col=1)

def _study_series_points(studies: Any) -> list[dict[str, Any]]:
    points = []
    for index, study in enumerate(studies):
        user_attrs = getattr(study, "user_attrs", {})
        score = user_attrs.get("incumbent_score")
        if score is None:
            continue

        name = getattr(study, "study_name", "")
        label = (
            name.split("_", 1)[0].upper()
            if name.startswith("s") and "_" in name
            else name or f"S{index}"
        )
        points.append({
            "score": float(score),
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

    points = _current_study_points(study)
    numbers = [point["trial_number"] for point in points]
    scores = [point["score"] for point in points]
    best_scores = [point["best_score"] for point in points]
    hover_params = [point["hover_params"] for point in points]

    figure.add_trace(
        go.Scatter(
            x=numbers,
            y=scores,
            mode="markers",
            name="Score",
            showlegend=False,
            marker=dict(color="#1f77b4"),
            customdata=hover_params,
            hovertemplate=(
                "Trial: %{x}<br>Score: %{y:.1f}"
                "%{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=numbers,
            y=best_scores,
            mode="lines",
            name="Best score",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )
    figure.update_xaxes(
        title_text="Trial",
        range=[0, max(1, target_trials)],
        row=2,
        col=1,
    )
    score_floor = min([0, *scores])
    figure.update_yaxes(
        title_text="Score",
        range=[score_floor - 10, 260],
        row=2,
        col=1,
    )


def _current_study_points(study: Any) -> list[dict[str, Any]]:
    trials = [
        trial for trial in study.trials
        if trial.state.name == "COMPLETE"
        and trial.value is not None
    ]
    points = []
    best = float("-inf")
    for trial in trials:
        score = float(trial.value)
        best = max(best, score)
        points.append({
            "trial_number": trial.number,
            "score": score,
            "best_score": best,
            "hover_params": "".join(
                f"<br>{name}: {value}" for name, value in trial.params.items()
            ),
        })
    return points


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
            hovertemplate="%{customdata}<br>Score: %{y:.1f}<extra></extra>",
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
            hovertemplate="Mean score: %{y:.1f}<extra></extra>",
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
    figure.update_yaxes(title_text="Score", row=2, col=2)

def _robustness_offsets(count: int) -> list[float]:
    if count == 1:
        return [0.0]
    center = (count - 1) / 2
    return [(index - center) * 0.06 for index in range(count)]
