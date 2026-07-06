"""Current-HPs panel."""

from typing import Any

from hpo.study_reporting import TrainingProgress


def current_params(
    incumbent_params: dict[str, Any], training_progress: TrainingProgress | None
) -> dict[str, Any]:
    if training_progress is None or training_progress.trial_params is None:
        return incumbent_params
    return training_progress.trial_params


def add_current_hps(
    figure: Any, params: dict[str, Any], study: Any, *, optimized_param_names: list[str] | None = None
) -> None:
    import plotly.graph_objects as go

    names = list(params)
    optimized_params = {name for trial in study.trials for name in trial.params}
    if optimized_param_names is not None:
        optimized_params.update(optimized_param_names)
    row_colors = ["#fff2cc" if name in optimized_params else "white" for name in names]
    figure.add_trace(
        go.Table(
            columnwidth=[1.5, 1.0],
            header=dict(
                values=["", ""],
                height=1,
                fill_color="rgba(0,0,0,0)",
                line_color="rgba(0,0,0,0)",
                font=dict(color="rgba(0,0,0,0)", size=1),
            ),
            cells=dict(
                values=[names, [_format_hp_value(name, params[name]) for name in names]],
                align=["left", "right"],
                fill_color=[row_colors, row_colors],
                line_color="white",
                height=22,
            ),
        ),
        row=1,
        col=1,
    )


def _format_hp_value(name: str, value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
