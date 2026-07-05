"""Study-series panel."""

from typing import Any


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
