"""Current-trial training panel."""

from typing import Any

from hpo.study_reporting import TrainingProgress

_ENV_LABELS = {
    "mercury": ("Mercury", "#4f4f4f"),
    "venus": ("Venus", "#f2c94c"),
    "earth": ("Earth", "#1f77d4"),
    "moon": ("Moon", "#809090"),
    "mars": ("Mars", "#ff5f0e"),
}
_FALLBACK_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)


def add_training_progress(
    figure: Any,
    progress: TrainingProgress,
    training_score_min: float | None,
) -> None:
    import plotly.graph_objects as go

    returns = [float(value) for value in progress.episode_returns]
    episodes = list(range(1, len(returns) + 1))
    x_range = [1, max(progress.target_episodes, len(returns), 1)]
    has_env_labels = bool(progress.episode_env_labels)
    figure.add_trace(
        go.Scatter(
            x=episodes,
            y=returns,
            mode="lines" if has_env_labels else "lines+markers",
            name="Episode return",
            showlegend=False,
            marker=dict(color="#1f77b4", size=5),
            line=dict(color="#9a9a9a" if has_env_labels else "#1f77b4", width=1),
            hovertemplate="Episode: %{x}<br>Return: %{y:.1f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    if has_env_labels:
        _add_env_label_traces(figure, progress, episodes, returns)
    if progress.episode_epsilons:
        epsilon_count = min(len(episodes), len(progress.episode_epsilons))
        figure.add_trace(
            go.Scatter(
                x=episodes[:epsilon_count],
                y=[float(value) for value in progress.episode_epsilons[:epsilon_count]],
                mode="lines",
                name="Epsilon",
                showlegend=False,
                line=dict(color="#2ca02c", dash="dot"),
                hovertemplate="Epsilon: %{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

    mean_label = "Mean: n/a"
    if progress.checkpoint_window is not None:
        mean_episodes, means = _trailing_means(returns, progress.checkpoint_window)
        if means:
            mean_label = (
                f"Current Mean: {means[-1]:.1f}"
                f" · Best Mean: {max(means):.1f}"
            )
            figure.add_trace(
                go.Scatter(
                    x=mean_episodes,
                    y=means,
                    mode="lines",
                    name=f"Mean ({progress.checkpoint_window} episodes)",
                    showlegend=False,
                    line=dict(color="red"),
                    hovertemplate="Mean: %{y:.1f}<extra></extra>",
                ),
                row=3,
                col=1,
                secondary_y=False,
            )

    reference_score = (
        progress.best_checkpoint_score
        if progress.best_checkpoint_score is not None
        else progress.checkpoint_min_score
    )
    if reference_score is not None:
        reference_name = (
            "Best checkpoint score"
            if progress.best_checkpoint_score is not None
            else "Checkpoint threshold"
        )
        figure.add_trace(
            go.Scatter(
                x=x_range,
                y=[reference_score, reference_score],
                mode="lines",
                name=reference_name,
                showlegend=False,
                line=dict(color="gray", dash="dash"),
                hovertemplate=f"{reference_name}: %{{y:.1f}}<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )

    figure.layout.annotations[4].text = (
        f"Trial: {progress.trial_number} · {mean_label}"
    )
    figure.update_xaxes(title_text="Training episode", range=x_range, row=3, col=1)
    score_values = [0, *returns]
    if reference_score is not None:
        score_values.append(reference_score)
    figure.update_yaxes(
        title_text="Gym score",
        range=[
            max(min(score_values) - 10, training_score_min)
            if training_score_min is not None
            else min(score_values) - 10,
            max(score_values) + 10,
        ],
        row=3,
        col=1,
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Epsilon",
        range=[0, 1],
        row=3,
        col=1,
        secondary_y=True,
    )


def _add_env_label_traces(
    figure: Any,
    progress: TrainingProgress,
    episodes: list[int],
    returns: list[float],
) -> None:
    import plotly.graph_objects as go

    seen_labels = list(dict.fromkeys(progress.episode_env_labels))
    labels = [
        *[label for label in _ENV_LABELS if label in seen_labels],
        *[label for label in seen_labels if label not in _ENV_LABELS],
    ]
    fallback_colors = {
        label: _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]
        for index, label in enumerate(labels)
    }
    for label in labels:
        label_episodes = [
            episode
            for episode, episode_label in zip(episodes, progress.episode_env_labels)
            if episode_label == label
        ]
        label_returns = [
            episode_return
            for episode_return, episode_label in zip(
                returns,
                progress.episode_env_labels,
            )
            if episode_label == label
        ]
        label_name = label if label is not None else "unknown"
        display_name = _display_name(label_name)
        color = _ENV_LABELS.get(label_name, (display_name, fallback_colors[label]))[1]
        figure.add_trace(
            go.Scatter(
                x=label_episodes,
                y=label_returns,
                mode="markers",
                name=display_name,
                showlegend=True,
                marker=dict(
                    color=color,
                    size=6,
                ),
                hovertemplate=(
                    f"{display_name}<br>"
                    "Episode: %{x}<br>Return: %{y:.1f}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
            secondary_y=False,
        )


def _display_name(label: str) -> str:
    return _ENV_LABELS.get(label, (label, ""))[0]


def _trailing_means(values: list[float], window: int) -> tuple[list[int], list[float]]:
    if window < 1 or len(values) < window:
        return [], []
    episodes = []
    means = []
    for end in range(window, len(values) + 1):
        episodes.append(end)
        window_values = values[end - window:end]
        means.append(sum(window_values) / len(window_values))
    return episodes, means
