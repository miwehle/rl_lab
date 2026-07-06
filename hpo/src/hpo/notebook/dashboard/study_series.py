"""Study-series panel."""

from typing import Any

from hpo.notebook.dashboard.style import NO_DATA_TEXT, set_empty_score_yaxis


def add_study_series(figure: Any, studies: Any) -> None:
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
            showlegend=False,
            line=dict(color="red"),
            marker=dict(color="black"),
        ),
        row=1,
        col=1,
    )
    x_range = [-0.5, max(0.5, len(points) - 0.5)]
    figure.update_xaxes(
        title_text="Study",
        tickmode="array",
        tickvals=x,
        ticktext=labels,
        range=x_range,
        row=1,
        col=1,
    )
    if not scores:
        figure.add_annotation(
            text=NO_DATA_TEXT,
            row=1,
            col=1,
            showarrow=False,
            font=dict(color="gray"),
        )
        set_empty_score_yaxis(figure, row=1, col=1)
    else:
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
