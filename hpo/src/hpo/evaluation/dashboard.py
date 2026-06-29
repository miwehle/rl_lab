"""Tell the story of an HPO study series in one notebook dashboard.

The dashboard is the visual interface between the human and the running HPO:
- Study Series shows the overall progress.
- Current HPs shows the running trial hyperparameters, or the incumbent otherwise.
- Study follows the current optimization.
- HP Robustness Evaluation confirms the best candidates at the end of each study.
- Current Trial Training shows live episode returns for the running trial, so the
  human at the dashboard can see how training is going before the trial finishes.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from hpo.checkpointing import trial_best_checkpoint_score
from hpo.study_reporting import (
    RobustnessProgress,
    StudySeriesReporter,
    TrainingProgress,
)


DashboardRenderMode = Literal["safe"]
_ENV_LABEL_COLORS = {
    "moon": "#809090",
    "mercury": "#4f4f4f",
    "mars": "#ff5f0e",
    "earth": "#1f77d4",
    "venus": "#f2c94c",
}
_ENV_LABEL_ORDER = ("mercury", "venus", "earth", "moon", "mars")
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


def build_dashboard(
    *,
    study: Any,
    target_trials: int,
    studies: Any,
    incumbent_params: dict[str, Any],
    robustness_progress: RobustnessProgress | None = None,
    training_progress: TrainingProgress | None = None,
    training_score_min: float | None = -500.0,
) -> Any:
    """Build the study-series dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{}, {"type": "domain"}],
            [{}, {}],
            [{"colspan": 2, "secondary_y": True}, None],
        ],
        row_heights=[0.36, 0.32, 0.32],
        vertical_spacing=0.11,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Study Series",
            "Current HPs",
            f"Study: {_study_title(study)}",
            "HP Robustness Evaluation",
            "Current Trial Training",
        ),
    )
    _add_study_series(figure, studies)
    _add_current_hps(
        figure,
        _current_params(incumbent_params, training_progress),
        study,
        optimized_param_names=(
            training_progress.optimized_param_names
            if training_progress is not None
            else None
        ),
    )
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

    if training_progress is None:
        figure.add_annotation(
            text="Waiting for trial training",
            row=3,
            col=1,
            showarrow=False,
            font=dict(color="gray"),
        )
    else:
        _add_training_progress(figure, training_progress, training_score_min)

    _style_dashboard(figure)
    return figure


def _style_dashboard(figure: Any) -> None:
    figure.update_layout(
        width=1200,
        height=850,
        margin=dict(l=70, r=70, t=55, b=55),
        legend=dict(
            orientation="h",
            x=0.01,
            y=1.04,
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


@dataclass
class DashboardContext:
    study: Any
    target_trials: int
    robustness_progress: RobustnessProgress | None = None


@dataclass
class StudySeriesContext:
    studies: list[Any]
    incumbent_params: dict[str, Any]


class Dashboard(StudySeriesReporter):
    """Report study-series progress through the notebook dashboard.

    render_mode="safe" clears and redisplays the whole dashboard. This is robust
    in notebooks and Colab, but can visibly flicker during live training.
    training_update_interval_seconds throttles live training updates.
    """

    def __init__(
        self,
        *,
        render_mode: DashboardRenderMode = "safe",
        training_update_interval_seconds: float = 5.0,
        training_score_min: float | None = -500.0,
    ) -> None:
        if render_mode != "safe":
            raise ValueError(f"unsupported dashboard render_mode: {render_mode}")
        self.render_mode = render_mode
        self.training_update_interval_seconds = training_update_interval_seconds
        self.training_score_min = training_score_min
        self._last_training_update = 0.0
        self._context: DashboardContext | None = None
        self._series: StudySeriesContext | None = None

    def set_study_series_context(
        self,
        *,
        studies: list[Any],
        incumbent_params: dict[str, Any],
    ) -> None:
        self._series = StudySeriesContext(
            studies=studies,
            incumbent_params=incumbent_params,
        )

    def report_optimization(
        self,
        study: Any,
        *,
        target_trials: int,
    ) -> None:
        self._context = DashboardContext(
            study=study,
            target_trials=target_trials,
        )
        self._show()

    def report_robustness_evaluation(
        self,
        progress: RobustnessProgress,
    ) -> None:
        if self._context is None:
            return
        self._context = DashboardContext(
            study=self._context.study,
            target_trials=self._context.target_trials,
            robustness_progress=progress,
        )
        self._show()

    def report_training_progress(self, progress: TrainingProgress) -> None:
        if self._context is None:
            return
        now = perf_counter()
        is_final = len(progress.episode_returns) >= progress.target_episodes
        if (
            not is_final
            and now - self._last_training_update < self.training_update_interval_seconds
        ):
            return
        self._last_training_update = now
        self._show(training_progress=progress)

    def _show(self, training_progress: TrainingProgress | None = None) -> None:
        if self._context is None or self._series is None:
            return
        _clear_output(wait=True)
        _display(
            build_dashboard(
                study=self._context.study,
                target_trials=self._context.target_trials,
                studies=self._series.studies,
                incumbent_params=self._series.incumbent_params,
                robustness_progress=self._context.robustness_progress,
                training_progress=training_progress,
                training_score_min=self.training_score_min,
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


def _current_params(
    incumbent_params: dict[str, Any],
    training_progress: TrainingProgress | None,
) -> dict[str, Any]:
    if training_progress is None or training_progress.trial_params is None:
        return incumbent_params
    return training_progress.trial_params


def _add_current_hps(
    figure: Any,
    params: dict[str, Any],
    study: Any,
    *,
    optimized_param_names: list[str] | None = None,
) -> None:
    import plotly.graph_objects as go

    names = list(params)
    optimized_params = {
        name
        for trial in study.trials
        for name in trial.params
    }
    if optimized_param_names is not None:
        optimized_params.update(optimized_param_names)
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
                    [_format_hp_value(name, params[name]) for name in names],
                ],
                align=["left", "right"],
                fill_color=[row_colors, row_colors],
                height=22,
            ),
        ),
        row=1,
        col=2,
    )


def _format_hp_value(name: str, value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


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
            marker=dict(color="black"),
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
        score = trial_best_checkpoint_score(trial)
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


def _add_training_progress(
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
        seen_labels = list(dict.fromkeys(progress.episode_env_labels))
        labels = [
            *[label for label in _ENV_LABEL_ORDER if label in seen_labels],
            *[label for label in seen_labels if label not in _ENV_LABEL_ORDER],
        ]
        fallback_colors = {
            label: _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]
            for index, label in enumerate(labels)
        }
        for label in labels:
            label_episodes = [
                episode
                for episode, episode_label in zip(
                    episodes,
                    progress.episode_env_labels,
                )
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
            figure.add_trace(
                go.Scatter(
                    x=label_episodes,
                    y=label_returns,
                    mode="markers",
                    name=label_name,
                    showlegend=True,
                    marker=dict(
                        color=_ENV_LABEL_COLORS.get(label_name, fallback_colors[label]),
                        size=6,
                    ),
                    hovertemplate=(
                        f"{label_name}<br>"
                        "Episode: %{x}<br>Return: %{y:.1f}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
                secondary_y=False,
            )
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
                f"Mean ({progress.checkpoint_window} episodes): {means[-1]:.1f}"
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
        f"Current Trial Training - Trial {progress.trial_number} - {mean_label}"
    )
    figure.update_xaxes(
        title_text="Episode",
        range=x_range,
        row=3,
        col=1,
    )
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


def _trailing_means(values: list[float], window: int) -> tuple[list[int], list[float]]:
    if window < 1 or len(values) < window:
        return [], []
    episodes = []
    means = []
    for end in range(window, len(values) + 1):
        episodes.append(end)
        window_values = values[end - window:end]
        means.append(sum(window_values) / window)
    return episodes, means

def _robustness_offsets(count: int) -> list[float]:
    if count == 1:
        return [0.0]
    center = (count - 1) / 2
    return [(index - center) * 0.06 for index in range(count)]
