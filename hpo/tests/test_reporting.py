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


def test_show_lander_live_progress_updates_fixed_dashboard(monkeypatch) -> None:
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
    dashboard = reporting._Dashboard(
        container="dashboard",
        lander_history="lh",
        optimization_history="oh",
        podium="podium",
    )
    shown = []
    cleared = []
    monkeypatch.setattr(reporting, "_get_dashboard", lambda _studies: dashboard)
    monkeypatch.setattr(
        reporting,
        "_show_in",
        lambda panel, value, **kwargs: shown.append((panel, value, kwargs)),
    )
    monkeypatch.setattr(
        reporting,
        "_clear_panel",
        lambda panel, **kwargs: cleared.append((panel, kwargs)),
    )
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

    assert shown[0][0] == "lh"
    assert shown[0][1].axes[0].get_ylabel() == "Gym score"
    assert not plt.fignum_exists(shown[0][1].number)
    assert shown[1] == (
        "oh",
        ("oh", study, 40),
        {"heading": "Study: Unnamed"},
    )
    assert cleared == [(
        "podium",
        {
            "heading": "Podium",
            "message": "Waiting for robustness evaluation",
        },
    )]


def test_dashboard_is_reused_within_one_study_series(monkeypatch) -> None:
    first = FakeStudy(trials=[])
    second = FakeStudy(trials=[])
    dashboards = [
        reporting._Dashboard("dashboard-1", "lh-1", "oh-1", "p-1"),
        reporting._Dashboard("dashboard-2", "lh-2", "oh-2", "p-2"),
    ]
    created = []
    displayed = []
    cleared = []
    monkeypatch.setattr(reporting, "_dashboard", None)
    monkeypatch.setattr(reporting, "_dashboard_series", None)
    monkeypatch.setattr(
        reporting,
        "_create_dashboard",
        lambda: created.append(None) or dashboards[len(created) - 1],
    )
    monkeypatch.setattr(reporting, "_display", displayed.append)
    monkeypatch.setattr(
        reporting,
        "_clear_output",
        lambda **kwargs: cleared.append(kwargs),
    )

    assert reporting._get_dashboard([first]) is dashboards[0]
    assert reporting._get_dashboard([first, second]) is dashboards[0]
    assert reporting._get_dashboard([second]) is dashboards[1]

    assert len(created) == 2
    assert displayed == ["dashboard-1", "dashboard-2"]
    assert cleared == [{"wait": True}, {"wait": True}]


def test_show_in_embeds_plotly_as_colab_html() -> None:
    import ipywidgets as widgets
    import plotly.graph_objects as go

    panel = widgets.Output()
    reporting._show_in(
        panel,
        go.Figure(go.Scatter(x=[1], y=[2])),
        heading="Study: Test",
    )

    assert panel.outputs[0]["text"] == "Study: Test\n"
    assert "text/html" in panel.outputs[1]["data"]


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

def test_plot_robustness_progress_marks_candidate_states() -> None:
    fig = reporting.plot_robustness_progress([-1.0, -1.5, -2.0], 2)
    ax = fig.axes[0]

    assert ax.get_ylabel() == "Mean QE score"
    assert ax.get_title() == "Robustness Candidates"
    assert [patch.get_y() + patch.get_height() for patch in ax.patches] == [
        pytest.approx(-1.0),
        pytest.approx(-1.5),
        pytest.approx(-2.0),
    ]
    from matplotlib.colors import to_rgba

    assert [patch.get_facecolor() for patch in ax.patches] == [
        to_rgba("tab:blue"),
        to_rgba("tab:orange"),
        to_rgba("lightgray"),
    ]


def test_show_robustness_progress_replaces_oh_with_podium(monkeypatch) -> None:
    study = FakeStudy(trials=[], study_name="s1_qe_update_economy")
    dashboard = reporting._Dashboard(
        container="dashboard",
        lander_history="lh",
        optimization_history="oh",
        podium="podium",
    )
    shown = []
    cleared = []
    monkeypatch.setattr(reporting, "_get_dashboard", lambda _studies: dashboard)
    monkeypatch.setattr(
        reporting,
        "_show_in",
        lambda panel, value, **kwargs: shown.append((panel, value, kwargs)),
    )
    monkeypatch.setattr(
        reporting,
        "_clear_panel",
        lambda panel, **kwargs: cleared.append((panel, kwargs)),
    )

    reporting.show_robustness_progress(
        study,
        lander_studies=[],
        candidate_index=2,
        candidate_count=3,
        seed_index=1,
        seed_count=1,
        candidate_scores=[-1.0, -1.5, -2.0],
    )

    assert shown[0][0] == "lh"
    assert shown[0][1].axes[0].get_title() == "Lander History"
    assert cleared == [(
        "oh",
        {
            "heading": "Study: S1 Update Economy",
            "message": "Optimization complete",
        },
    )]
    assert shown[1][0] == "podium"
    assert shown[1][1].axes[0].get_title() == "Robustness Candidates"
    assert shown[1][2] == {
        "heading": (
            "Study: S1 Update Economy | Robustness evaluation | "
            "Candidate 2/3 | Seed 1/1"
        ),
    }
    assert all(not plt.fignum_exists(call[1].number) for call in shown)
