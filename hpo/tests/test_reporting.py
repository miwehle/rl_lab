from dataclasses import dataclass, field
import sys
from types import ModuleType

import matplotlib.pyplot as plt
import pytest

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
    params: dict = field(default_factory=dict)
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
    assert fig.get_size_inches().tolist() == [11.0, 2.7]
    legend = gym_ax.get_legend()
    assert legend._loc == 3
    assert legend._ncols == 4


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


def test_show_lander_live_progress_displays_one_dashboard(monkeypatch) -> None:
    study = FakeStudy(
        trials=[FakeTrial(
            number=0,
            value=1,
            user_attrs={"wall_time_seconds": 60, "gym_score": 180},
        )],
        user_attrs={
            "robust_best_gym_score": 180,
            "robust_best_objective_score": -1.0,
            "robust_best_training_effort": 1.0,
        },
    )
    displayed = []
    cleared = []
    monkeypatch.setattr(
        reporting,
        "_dashboard_figure",
        lambda **kwargs: ("dashboard", kwargs),
    )
    monkeypatch.setattr(reporting, "_display", displayed.append)
    monkeypatch.setattr(
        reporting, "_clear_output", lambda **kwargs: cleared.append(kwargs)
    )

    show_lander_live_progress(
        study,
        target_trials=40,
        lander_studies=[study],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
    )

    assert cleared == [{"wait": True}]
    assert displayed == [(
        "dashboard",
        {
            "study": study,
            "target_trials": 40,
            "lander_studies": [study],
            "incumbent_params": {
                "learning_rate": 0.001,
                "gamma": 0.99,
            },
        },
    )]


def test_dashboard_contains_fixed_three_panel_layout() -> None:
    study = FakeStudy(
        trials=[FakeTrial(0, -2.0, {"gym_score": 50.0})],
        study_name="s1_qe_update_economy",
    )
    baseline = FakeStudy(
        trials=[],
        study_name="s0_qe_baseline",
        user_attrs={
            "robust_best_gym_score": 30.0,
            "robust_best_objective_score": -2.4,
            "robust_best_training_effort": 1.0,
        },
    )

    figure = reporting._dashboard_figure(
        study=study,
        target_trials=40,
        lander_studies=[baseline, study],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
    )

    assert figure.layout.width == 1200
    assert figure.layout.height == 650
    assert [annotation.text for annotation in figure.layout.annotations[:4]] == [
        "Study Series",
        "Best HPs (Current Incumbent)",
        "Study: S1 Update Economy",
        "HP Robustness Evaluation",
    ]
    table = next(trace for trace in figure.data if trace.type == "table")
    assert list(table.cells.values[0]) == ["learning_rate", "gamma"]
    assert any(
        annotation.text == "Waiting for robustness evaluation"
        for annotation in figure.layout.annotations
    )
    assert {trace.name for trace in figure.data} >= {
        "Gym score",
        "QE score",
        "Best QE score",
        "Gym 200",
        "Gym 250",
    }
    assert {
        trace.name for trace in figure.data
        if getattr(trace, "showlegend", False) is not False
        and trace.name is not None
    } == {"Gym score", "QE score", "Best QE score"}


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
        trials=[FakeTrial(
            0,
            -2.0,
            {"gym_score": 50.0},
            {"learning_rate": 0.001, "batch_size": 512},
        )],
    )

    result = reporting._optimization_history_figure(study, 40)

    assert result is figure
    assert figure.added_traces[0]["name"] == "Gym score"
    assert figure.added_traces[0]["yaxis"] == "y"
    assert figure.added_traces[0]["customdata"] == [
        "<br>learning_rate: 0.001<br>batch_size: 512"
    ]
    assert "Gym score" in figure.added_traces[0]["hovertemplate"]
    assert figure.data[0].updates["name"] == "QE score"
    assert figure.data[0].updates["yaxis"] == "y2"
    assert figure.data[0].updates["customdata"] == [
        "<br>learning_rate: 0.001<br>batch_size: 512"
    ]
    assert "QE score" in figure.data[0].updates["hovertemplate"]
    assert figure.data[1].updates["name"] == "Best QE score"
    assert figure.data[1].updates["line_color"] == "red"
    assert "Best QE score" in figure.data[1].updates["hovertemplate"]
    assert figure.layout["width"] == 535
    assert figure.layout["height"] == 315
    assert figure.layout["showlegend"] is False
    assert figure.layout["yaxis"]["title"] == "Gym score"
    assert figure.layout["yaxis2"]["title"] == "QE score"
    assert [line["y"] for line in figure.hlines] == [200, 250]

def test_robustness_plot_shows_seed_scores_and_means() -> None:
    study = FakeStudy(trials=[], study_name="s1_qe_update_economy")
    figure = reporting._dashboard_figure(
        study=study,
        target_trials=40,
        lander_studies=[],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
        candidate_index=2,
        candidate_count=3,
        seed_index=2,
        seed_count=4,
        candidate_seed_scores=[
            [-1.0, -1.2, -0.8],
            [-1.5, -1.4],
            [-2.0],
        ],
    )

    seed_trace, mean_trace = figure.data[-2:]
    assert list(seed_trace.y) == [-1.0, -1.2, -0.8, -1.5, -1.4, -2.0]
    assert list(seed_trace.customdata) == [
        "Optimize trial",
        "Extra seed 1",
        "Extra seed 2",
        "Optimize trial",
        "Extra seed 1",
        "Optimize trial",
    ]
    assert list(mean_trace.y) == pytest.approx([-1.0, -1.45, -2.0])
    assert mean_trace.marker.symbol == "diamond"
    assert any(
        annotation.text == "Waiting for optimization"
        for annotation in figure.layout.annotations
    )


def test_show_robustness_progress_replaces_oh_with_podium(monkeypatch) -> None:
    study = FakeStudy(trials=[], study_name="s1_qe_update_economy")
    displayed = []
    cleared = []
    monkeypatch.setattr(
        reporting,
        "_dashboard_figure",
        lambda **kwargs: ("dashboard", kwargs),
    )
    monkeypatch.setattr(reporting, "_display", displayed.append)
    monkeypatch.setattr(
        reporting, "_clear_output", lambda **kwargs: cleared.append(kwargs)
    )

    reporting.show_robustness_progress(
        study,
        lander_studies=[],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
        candidate_index=2,
        candidate_count=3,
        seed_index=1,
        seed_count=1,
        candidate_seed_scores=[[-1.0], [-1.5], [-2.0]],
    )

    assert cleared == [{"wait": True}]
    assert displayed == [(
        "dashboard",
        {
            "study": study,
            "target_trials": 0,
            "lander_studies": [],
            "incumbent_params": {
                "learning_rate": 0.001,
                "gamma": 0.99,
            },
            "candidate_index": 2,
            "candidate_count": 3,
            "seed_index": 1,
            "seed_count": 1,
            "candidate_seed_scores": [[-1.0], [-1.5], [-2.0]],
        },
    )]
