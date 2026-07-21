"""Comparison plots for distilled students."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def plot_score_quantiles(teacher_summary: dict, student_summary: dict, worlds: Sequence[str] | None = None):
    """Plot teacher/student score distributions per world plus all worlds."""
    import plotly.graph_objects as go

    labels = _labels(teacher_summary, student_summary, worlds)
    figure = go.Figure()
    _add_quantiles(figure, teacher_summary, labels, name="Teacher", offset=-0.16, color="#1f77b4")
    _add_quantiles(figure, student_summary, labels, name="Student", offset=0.16, color="#ff7f0e")
    figure.update_layout(
        title="Teacher vs Student Score Distributions",
        xaxis_title="Score",
        yaxis=dict(
            title="World",
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            autorange="reversed",
        ),
        height=max(360, 70 * len(labels)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure


def plot_score_gaps(teacher_summary: dict, student_summary: dict, worlds: Sequence[str] | None = None):
    """Plot student minus teacher mean score gaps per world plus all worlds."""
    import plotly.graph_objects as go

    labels = _labels(teacher_summary, student_summary, worlds)
    teacher = _summary_by_label(teacher_summary, labels)
    student = _summary_by_label(student_summary, labels)
    gaps = [student[label]["mean"] - teacher[label]["mean"] for label in labels]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=gaps,
            marker_color=["#2ca02c" if gap >= 0 else "#d62728" for gap in gaps],
            customdata=[
                [teacher[label]["mean"], student[label]["mean"], gap]
                for label, gap in zip(labels, gaps, strict=True)
            ],
            hovertemplate=(
                "%{x}<br>Teacher mean: %{customdata[0]:.1f}"
                "<br>Student mean: %{customdata[1]:.1f}"
                "<br>Gap: %{customdata[2]:+.1f}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_color="black", line_width=1)
    figure.update_layout(title="Student - Teacher Mean Score Gap", xaxis_title="World", yaxis_title="Score gap")
    return figure


def _add_quantiles(figure, summary: dict, labels: Sequence[str], *, name: str, offset: float, color: str) -> None:
    values = _summary_by_label(summary, labels)
    y = [index + offset for index in range(len(labels))]
    for interval, low, high, width, opacity in [
        ("q05..q95", "q05", "q95", 7, 0.35),
        ("q25..q75", "q25", "q75", 15, 0.80),
    ]:
        figure.add_trace(
            {
                "type": "scatter",
                "x": _interval_x(values, labels, low, high),
                "y": _interval_y(y),
                "mode": "lines",
                "line": {"color": color, "width": width},
                "opacity": opacity,
                "name": f"{name} {interval}",
                "legendgroup": name,
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )
    figure.add_trace(
        {
            "type": "scatter",
            "x": [values[label]["median"] for label in labels],
            "y": y,
            "mode": "markers",
            "marker": {"color": "white", "line": {"color": color, "width": 2}, "size": 9},
            "name": f"{name} median",
            "legendgroup": name,
            "hovertemplate": f"{name}<br>%{{customdata}}<br>Median: %{{x:.1f}}<extra></extra>",
            "customdata": labels,
        }
    )
    figure.add_trace(
        {
            "type": "scatter",
            "x": [values[label]["mean"] for label in labels],
            "y": y,
            "mode": "markers",
            "marker": {"color": color, "symbol": "x", "size": 10},
            "name": f"{name} mean",
            "legendgroup": name,
            "hovertemplate": f"{name}<br>%{{customdata}}<br>Mean: %{{x:.1f}}<extra></extra>",
            "customdata": labels,
        }
    )


def _summary_by_label(summary: dict, labels: Sequence[str]) -> dict[str, dict[str, float]]:
    rows = summary["rows"]
    by_world = {label: _score_summary([row["score"] for row in rows if row["world"] == label]) for label in labels if label != "all"}
    by_world["all"] = _score_summary([row["score"] for row in rows])
    return by_world


def _score_summary(scores: Sequence[float]) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _labels(teacher_summary: dict, student_summary: dict, worlds: Sequence[str] | None) -> list[str]:
    available = {row["world"] for row in teacher_summary["rows"]} & {row["world"] for row in student_summary["rows"]}
    labels = list(worlds) if worlds is not None else [world for world in _ordered_worlds(teacher_summary) if world in available]
    return [label for label in labels if label in available] + ["all"]


def _ordered_worlds(summary: dict) -> list[str]:
    return list(dict.fromkeys(row["world"] for row in summary["rows"]))


def _interval_x(values: dict[str, dict[str, float]], labels: Sequence[str], low: str, high: str) -> list[float | None]:
    points = []
    for label in labels:
        points.extend([values[label][low], values[label][high], None])
    return points


def _interval_y(y: Sequence[float]) -> list[float | None]:
    points = []
    for value in y:
        points.extend([value, value, None])
    return points
