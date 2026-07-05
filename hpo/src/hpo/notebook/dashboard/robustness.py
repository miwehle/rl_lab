"""Robustness panel."""

from typing import Any


def add_robustness_evaluation(
    figure: Any,
    *,
    candidate_seed_scores: list[list[float]],
    candidate_index: int | None,
    first_score_label: str = "Optimize trial",
    extra_score_label: str = "Extra seed",
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
                first_score_label if seed == 0 else f"{extra_score_label} {seed}"
            )
            point_colors.append(
                "#1f77b4"
                if candidate < candidate_index
                else "#ff7f0e"
                if candidate == candidate_index
                else "lightgray"
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


def add_checkpoint_robustness_evaluation(
    figure: Any,
    *,
    checkpoint_summaries: list[dict[str, Any]],
) -> None:
    import plotly.graph_objects as go

    if not checkpoint_summaries:
        figure.add_annotation(
            text="Waiting for checkpoint robustness",
            row=2,
            col=2,
            showarrow=False,
            font=dict(color="gray"),
        )
        return

    labels = [_checkpoint_candidate_label(summary) for summary in checkpoint_summaries]
    intervals = [
        ("min..max", "min", "max", "lightgray", 2),
        ("q05..q95", "q05", "q95", "rgba(31, 119, 180, 0.35)", 6),
        ("q25..q75", "q25", "q75", "rgba(31, 119, 180, 0.85)", 12),
    ]
    for name, low_key, high_key, color, width in intervals:
        for index, (label, summary) in enumerate(
            zip(labels, checkpoint_summaries, strict=True)
        ):
            figure.add_trace(
                go.Scatter(
                    x=[summary[low_key], summary[high_key]],
                    y=[label, label],
                    mode="lines",
                    name=name,
                    showlegend=index == 0,
                    line=dict(color=color, width=width),
                    hovertemplate=(
                        f"{label}<br>{name}: "
                        f"{summary[low_key]:.1f}..{summary[high_key]:.1f}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=2,
            )

    figure.add_trace(
        go.Scatter(
            x=[summary["median"] for summary in checkpoint_summaries],
            y=labels,
            mode="markers",
            name="median",
            marker=dict(color="white", line=dict(color="black", width=1), size=8),
            hovertemplate="%{y}<br>Median: %{x:.1f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=[summary["mean"] for summary in checkpoint_summaries],
            y=labels,
            mode="markers",
            name="mean",
            marker=dict(color="red", symbol="x", size=9),
            hovertemplate="%{y}<br>Mean: %{x:.1f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    score_values = [
        float(summary[key])
        for summary in checkpoint_summaries
        for key in ("min", "max")
    ]
    figure.update_xaxes(
        title_text="Gym score",
        range=[min(score_values) - 10, max(score_values) + 10],
        row=2,
        col=2,
    )
    figure.update_yaxes(title_text="Checkpoint", row=2, col=2)


def _checkpoint_candidate_label(summary: dict[str, Any]) -> str:
    candidate = summary.get("candidate")
    trial_number = summary.get("trial_number")
    if candidate is None:
        return str(trial_number) if trial_number is not None else "checkpoint"
    if trial_number is None:
        return f"C{candidate}"
    return f"C{candidate} trial {trial_number}"


def _robustness_offsets(count: int) -> list[float]:
    if count == 1:
        return [0.0]
    center = (count - 1) / 2
    return [(index - center) * 0.06 for index in range(count)]
