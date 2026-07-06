"""Tell the story of an HPO study series in one notebook dashboard."""

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Any, Literal

from hpo.notebook.dashboard.current_hps import add_current_hps, current_params
from hpo.notebook.dashboard.current_study import add_current_study
from hpo.notebook.dashboard.robustness import (
    add_checkpoint_robustness_evaluation,
    add_robustness_evaluation,
)
from hpo.notebook.dashboard.study_series import add_study_series
from hpo.notebook.dashboard.training_progress import add_training_progress
from hpo.study_reporting import (
    RobustnessProgress,
    StudySeriesReporter,
    TrainingProgress,
)


DashboardRenderMode = Literal["safe"]
NO_DATA_TEXT = "No data yet"


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
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{}, {"type": "domain"}],
            [{}, {}],
            [{"colspan": 2, "secondary_y": True}, None],
        ],
        row_heights=[0.40, 0.30, 0.30],
        vertical_spacing=0.11,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Study Series",
            "Current HPs",
            f"Study: {_study_title(study)}",
            _robustness_panel_title(robustness_progress),
            "Current Trial Training",
        ),
    )
    add_study_series(figure, studies)
    add_current_hps(
        figure,
        current_params(incumbent_params, training_progress),
        study,
        optimized_param_names=(
            training_progress.optimized_param_names
            if training_progress is not None
            else None
        ),
    )
    stored_checkpoint_summaries = _stored_checkpoint_summaries(study)
    if robustness_progress is None:
        if not add_current_study(figure, study, target_trials):
            figure.add_annotation(
                text=NO_DATA_TEXT,
                row=2,
                col=1,
                showarrow=False,
                font=dict(color="gray"),
            )
        if stored_checkpoint_summaries:
            add_checkpoint_robustness_evaluation(
                figure,
                checkpoint_summaries=stored_checkpoint_summaries,
            )
        else:
            figure.add_annotation(
                text=NO_DATA_TEXT,
                row=2,
                col=2,
                showarrow=False,
                font=dict(color="gray"),
            )
            figure.update_xaxes(
                title_text="Checkpoint",
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["C1", "C2", "C3"],
                range=[-0.5, 2.5],
                row=2,
                col=2,
            )
            figure.update_yaxes(
                title_text="Score",
                range=[0, 250],
                row=2,
                col=2,
            )
    else:
        figure.add_annotation(
            text="Waiting for optimization",
            row=2,
            col=1,
            showarrow=False,
            font=dict(color="gray"),
        )
        if robustness_progress.checkpoint_summaries is None:
            add_robustness_evaluation(
                figure,
                candidate_seed_scores=robustness_progress.candidate_seed_scores,
                candidate_index=robustness_progress.candidate_index,
                first_score_label=robustness_progress.first_score_label,
                extra_score_label=robustness_progress.extra_score_label,
            )
        else:
            add_checkpoint_robustness_evaluation(
                figure,
                checkpoint_summaries=robustness_progress.checkpoint_summaries,
            )
        figure.layout.annotations[3].text = (
            f"{robustness_progress.title} - "
            f"Candidate {robustness_progress.candidate_index}/"
            f"{robustness_progress.candidate_count} - "
            f"{robustness_progress.step_label} {robustness_progress.seed_index}/"
            f"{robustness_progress.seed_count}"
        )

    if training_progress is None:
        figure.add_annotation(
            text=NO_DATA_TEXT,
            row=3,
            col=1,
            showarrow=False,
            font=dict(color="gray"),
        )
    else:
        add_training_progress(figure, training_progress, training_score_min)

    _style_dashboard(figure)
    if training_progress is None:
        _hide_waiting_panel_axes(figure, row=3, col=1, secondary_y=True)
    return figure


def _style_dashboard(figure: Any) -> None:
    figure.update_layout(
        width=1200,
        height=925,
        margin=dict(l=70, r=70, t=55, b=85),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.09,
            xanchor="center",
            yanchor="top",
        ),
        plot_bgcolor="white",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    figure.update_yaxes(showgrid=True, gridcolor="#e5e5e5")


def _hide_waiting_panel_axes(
    figure: Any,
    *,
    row: int,
    col: int,
    secondary_y: bool = False,
) -> None:
    figure.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=row,
        col=col,
    )
    _hide_waiting_panel_yaxis(figure, row=row, col=col, secondary_y=False)
    if secondary_y:
        _hide_waiting_panel_yaxis(figure, row=row, col=col, secondary_y=True)


def _hide_waiting_panel_yaxis(
    figure: Any,
    *,
    row: int,
    col: int,
    secondary_y: bool,
) -> None:
    figure.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=row,
        col=col,
        secondary_y=secondary_y,
    )


def _study_title(study: Any) -> str:
    parts = getattr(study, "study_name", "").split("_")
    if not parts or not parts[0]:
        return "Unnamed"
    return " ".join([parts[0].upper(), *parts[1:]]).replace("_", " ").title()


def _robustness_panel_title(progress: RobustnessProgress | None) -> str:
    if progress is None:
        return "Checkpoint Robustness"
    return progress.title


def _stored_checkpoint_summaries(study: Any) -> list[dict[str, Any]] | None:
    results = getattr(study, "user_attrs", {}).get("checkpoint_robustness")
    if results is None:
        return None
    if isinstance(results, str):
        results = json.loads(results)
    return [
        result["checkpoint_summary"]
        for result in results
        if "checkpoint_summary" in result
    ]


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
    """Report study-series progress through the notebook dashboard."""

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
