"""Current-study panel."""

from typing import Any

from hpo.checkpointing import trial_best_checkpoint_score
from hpo.notebook.dashboard.style import set_empty_score_yaxis


def add_current_study(figure: Any, study: Any, target_trials: int) -> bool:
    import plotly.graph_objects as go

    points = _current_study_points(study)
    if not points:
        set_empty_score_yaxis(figure, row=1, col=2)
        return False

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
            hovertemplate=("Trial: %{x}<br>Score: %{y:.1f}" "%{customdata}<extra></extra>"),
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=numbers,
            y=best_scores,
            mode="lines",
            name="Best score",
            showlegend=False,
            line=dict(color="red"),
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(
        title_text="Trial",
        tickmode="array",
        tickvals=_trial_tickvals(target_trials),
        range=[0, max(1, target_trials)],
        row=1,
        col=2,
    )
    score_floor = min([0, *scores])
    figure.update_yaxes(title_text="Score", range=[score_floor - 10, 260], row=1, col=2)
    return True


def _trial_tickvals(target_trials: int) -> list[int]:
    end = max(1, target_trials)
    step = max(1, (end + 9) // 10)
    tickvals = list(range(0, end + 1, step))
    if tickvals[-1] != end:
        tickvals.append(end)
    return tickvals


def _current_study_points(study: Any) -> list[dict[str, Any]]:
    trials = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    points = []
    best = float("-inf")
    for trial in trials:
        score = trial_best_checkpoint_score(trial)
        best = max(best, score)
        points.append(
            {
                "trial_number": trial.number,
                "score": score,
                "best_score": best,
                "hover_params": "".join(f"<br>{name}: {value}" for name, value in trial.params.items()),
            }
        )
    return points
