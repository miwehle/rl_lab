from dataclasses import dataclass, field
import sys
from types import ModuleType

import matplotlib.pyplot as plt

from hpo.evaluation import reporting
from hpo.evaluation.reporting import plot_lander_progress, show_lander_live_progress


@dataclass
class FakeState:
    name: str


@dataclass
class FakeTrial:
    number: int
    value: float
    user_attrs: dict
    state: FakeState = field(default_factory=lambda: FakeState("COMPLETE"))


@dataclass
class FakeStudy:
    trials: list[FakeTrial]
    study_name: str = ""
    user_attrs: dict = field(default_factory=dict)


def test_plot_lander_progress_uses_robust_effort_and_gym_score() -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                number=1,
                value=10,
                user_attrs={"wall_time_seconds": 120, "gym_score": 210},
            ),
            FakeTrial(
                number=0,
                value=20,
                user_attrs={"wall_time_seconds": 60, "gym_score": 180},
            ),
            FakeTrial(
                number=2,
                value=30,
                user_attrs={"wall_time_seconds": 999, "other": 1},
            ),
        ],
        study_name="s1_update_economy",
        user_attrs={
            "robust_best_gym_score": 215,
            "robust_best_objective_score": -0.5,
            "robust_best_training_effort": 0.8,
        },
    )

    fig = plot_lander_progress(study)
    gym_ax, qe_ax = fig.axes

    assert list(gym_ax.lines[0].get_xdata()) == [0.8]
    assert list(gym_ax.lines[0].get_ydata()) == [215.0]
    assert list(qe_ax.lines[0].get_ydata()) == [-0.5]
    assert gym_ax.get_xlabel() == "Training effort relative to S0"
    assert gym_ax.get_ylabel() == "Gym score"
    assert qe_ax.get_ylabel() == "QE score"
    assert [line.get_ydata()[0] for line in gym_ax.lines[1:]] == [200, 250]


def test_plot_lander_progress_uses_one_point_per_study() -> None:
    studies = [
        FakeStudy(
            trials=[
                FakeTrial(
                    number=0,
                    value=1,
                    user_attrs={"wall_time_seconds": 60, "gym_score": 180},
                ),
            ],
            study_name="s0_baseline",
            user_attrs={
                "robust_best_gym_score": 180,
                "robust_best_objective_score": -1.0,
                "robust_best_training_effort": 1.0,
            },
        ),
        FakeStudy(
            trials=[
                FakeTrial(
                    number=0,
                    value=1,
                    user_attrs={"wall_time_seconds": 120, "gym_score": 210},
                ),
            ],
            study_name="s1_update_economy",
            user_attrs={
                "robust_best_gym_score": 210,
                "robust_best_objective_score": 0.1,
                "robust_best_training_effort": 0.9,
            },
        ),
    ]

    fig = plot_lander_progress(studies)
    ax = fig.axes[0]

    assert list(ax.lines[0].get_xdata()) == [1.0, 0.9]
    assert list(ax.lines[0].get_ydata()) == [180.0, 210.0]
    assert list(fig.axes[1].lines[0].get_ydata()) == [-1.0, 0.1]


def test_plot_lander_progress_skips_studies_without_robust_result() -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                number=0,
                value=1,
                user_attrs={"wall_time_seconds": 60, "gym_score": 180},
            ),
        ]
    )

    fig = plot_lander_progress(study)
    ax = fig.axes[0]

    assert list(ax.lines[0].get_xdata()) == []
    assert list(ax.lines[0].get_ydata()) == []


def test_show_lander_live_progress_displays_params_and_closes_lander_figure(
    monkeypatch,
) -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                number=0,
                value=1,
                user_attrs={"wall_time_seconds": 60, "gym_score": 180},
            ),
        ],
        user_attrs={
            "robust_best_gym_score": 180,
            "robust_best_objective_score": -1.0,
            "robust_best_training_effort": 1.0,
            "robust_best_params": {"learning_rate": 0.001},
        },
    )
    displayed = []
    clear_calls = []
    monkeypatch.setattr(
        reporting,
        "_clear_output",
        lambda **kwargs: clear_calls.append(kwargs),
    )
    monkeypatch.setattr(reporting, "_display", displayed.append)
    monkeypatch.setattr(
        reporting,
        "_optimization_history_figure",
        lambda current_study, target_trials: ("oh", current_study, target_trials),
    )

    show_lander_live_progress(
        study,
        target_trials=40,
        lander_studies=[study],
    )

    assert clear_calls == [{"wait": True}]
    assert len(displayed) == 3
    assert displayed[0].axes[0].get_ylabel() == "Gym score"
    assert displayed[0].axes[1].get_ylabel() == "QE score"
    assert not plt.fignum_exists(displayed[0].number)
    assert displayed[1] == {"learning_rate": 0.001}
    assert displayed[2] == ("oh", study, 40)

def test_optimization_history_uses_consistent_score_axes(monkeypatch) -> None:
    class FakeTrace:
        def __init__(self) -> None:
            self.updates = {}

        def update(self, **kwargs) -> None:
            self.updates.update(kwargs)

    class FakeFigure:
        def __init__(self) -> None:
            self.data = [FakeTrace(), FakeTrace()]
            self.added_traces = []
            self.hlines = []
            self.layout = {}
            self.xaxes = {}

        def add_trace(self, trace) -> None:
            self.added_traces.append(trace)

        def add_hline(self, **kwargs) -> None:
            self.hlines.append(kwargs)

        def update_layout(self, **kwargs) -> None:
            self.layout.update(kwargs)

        def update_xaxes(self, **kwargs) -> None:
            self.xaxes.update(kwargs)

    figure = FakeFigure()
    graph_objects = ModuleType("plotly.graph_objects")
    graph_objects.Scatter = lambda **kwargs: kwargs
    plotly = ModuleType("plotly")
    plotly.graph_objects = graph_objects
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", graph_objects)
    monkeypatch.setattr(reporting, "_plot_optimization_history", lambda _study: figure)
    study = FakeStudy(
        trials=[FakeTrial(0, -2.0, {"gym_score": 50.0})],
    )

    result = reporting._optimization_history_figure(study, 40)

    assert result is figure
    assert figure.added_traces[0]["name"] == "Gym score"
    assert figure.added_traces[0]["yaxis"] == "y"
    assert figure.data[0].updates["name"] == "QE score"
    assert figure.data[0].updates["yaxis"] == "y2"
    assert figure.data[1].updates["name"] == "Best QE score"
    assert figure.data[1].updates["line_color"] == "red"
    assert figure.layout["yaxis"]["title"] == "Gym score"
    assert figure.layout["yaxis2"]["title"] == "QE score"
    assert [line["y"] for line in figure.hlines] == [200, 250]
