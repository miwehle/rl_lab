from dataclasses import dataclass, field

import pytest

from hpo.evaluation import dashboard
from hpo.evaluation.dashboard import Dashboard
from hpo.study_reporting import RobustnessProgress


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


def test_show_dashboard_during_optimization_displays_one_dashboard(monkeypatch) -> None:
    study = FakeStudy(
        trials=[FakeTrial(
            number=0,
            value=180,
            user_attrs={"wall_time_seconds": 60},
        )],
        user_attrs={"incumbent_score": 180},
    )
    displayed = []
    cleared = []
    monkeypatch.setattr(
        dashboard,
        "build_dashboard",
        lambda **kwargs: ("dashboard", kwargs),
    )
    monkeypatch.setattr(dashboard, "_display", displayed.append)
    monkeypatch.setattr(
        dashboard, "_clear_output", lambda **kwargs: cleared.append(kwargs)
    )

    Dashboard().report_optimization(
        study,
        target_trials=40,
        studies=[study],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
    )

    assert cleared == [{"wait": True}]
    assert displayed == [(
        "dashboard",
        {
            "study": study,
            "target_trials": 40,
            "studies": [study],
            "incumbent_params": {
                "learning_rate": 0.001,
                "gamma": 0.99,
            },
        },
    )]


def test_dashboard_contains_fixed_four_panel_layout() -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                0,
                50.0,
                {},
                params={"learning_rate": 0.001},
            )
        ],
        study_name="s1_update_economy",
    )
    baseline = FakeStudy(
        trials=[],
        study_name="s1_flight_hours",
        user_attrs={"incumbent_score": 30.0},
    )

    figure = dashboard.build_dashboard(
        study=study,
        target_trials=40,
        studies=[baseline, study],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
    )

    assert figure.layout.width == 1200
    assert figure.layout.height == 650
    assert [annotation.text for annotation in figure.layout.annotations[:4]] == [
        "Study Series",
        "Best HPs",
        "Study: S1 Update Economy",
        "HP Robustness Evaluation",
    ]
    table = next(trace for trace in figure.data if trace.type == "table")
    assert list(table.cells.values[0]) == ["learning_rate", "gamma"]
    assert list(table.cells.fill.color[0]) == ["#fff2cc", "white"]
    assert list(table.cells.fill.color[1]) == ["#fff2cc", "white"]
    study_score_trace = next(
        trace
        for trace in figure.data
        if trace.name == "Score" and trace.xaxis == "x2"
    )
    assert list(study_score_trace.y) == [50.0]
    assert list(figure.layout.yaxis2.range) == [-10, 260]
    assert any(
        annotation.text == "Waiting for robustness evaluation"
        for annotation in figure.layout.annotations
    )
    assert {trace.name for trace in figure.data} >= {
        "Score",
        "Best score",
        "Score 200",
        "Score 250",
    }
    assert {
        trace.name for trace in figure.data
        if getattr(trace, "showlegend", False) is not False
        and trace.name is not None
    } == {"Score", "Best score"}


def test_robustness_plot_shows_seed_scores_and_means() -> None:
    study = FakeStudy(trials=[], study_name="s1_update_economy")
    figure = dashboard.build_dashboard(
        study=study,
        target_trials=40,
        studies=[],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
        robustness_progress=RobustnessProgress(
            candidate_index=2,
            candidate_count=3,
            seed_index=2,
            seed_count=4,
            candidate_seed_scores=[
                [100.0, 120.0, 80.0],
                [90.0, 110.0],
                [70.0],
            ],
        ),
    )

    seed_trace, mean_trace = figure.data[-2:]
    assert list(seed_trace.y) == [100.0, 120.0, 80.0, 90.0, 110.0, 70.0]
    assert list(seed_trace.customdata) == [
        "Optimize trial",
        "Extra seed 1",
        "Extra seed 2",
        "Optimize trial",
        "Extra seed 1",
        "Optimize trial",
    ]
    assert list(mean_trace.y) == pytest.approx([100.0, 100.0, 70.0])
    assert mean_trace.marker.symbol == "diamond"
    assert figure.layout.xaxis3.title.text == "Candidate"
    assert list(figure.layout.xaxis3.ticktext) == [1, 2, 3]
    assert any(
        annotation.text == "Waiting for optimization"
        for annotation in figure.layout.annotations
    )


def test_show_dashboard_during_robustness_evaluation_replaces_oh(monkeypatch) -> None:
    study = FakeStudy(trials=[], study_name="s1_update_economy")
    displayed = []
    cleared = []
    monkeypatch.setattr(
        dashboard,
        "build_dashboard",
        lambda **kwargs: ("dashboard", kwargs),
    )
    monkeypatch.setattr(dashboard, "_display", displayed.append)
    monkeypatch.setattr(
        dashboard, "_clear_output", lambda **kwargs: cleared.append(kwargs)
    )

    Dashboard().report_robustness_evaluation(
        study,
        studies=[],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
        progress=RobustnessProgress(
            candidate_index=2,
            candidate_count=3,
            seed_index=1,
            seed_count=1,
            candidate_seed_scores=[[-1.0], [-1.5], [-2.0]],
        ),
    )

    assert cleared == [{"wait": True}]
    assert displayed == [(
        "dashboard",
        {
            "study": study,
            "target_trials": 0,
            "studies": [],
            "incumbent_params": {
                "learning_rate": 0.001,
                "gamma": 0.99,
            },
            "robustness_progress": RobustnessProgress(
                candidate_index=2,
                candidate_count=3,
                seed_index=1,
                seed_count=1,
                candidate_seed_scores=[[-1.0], [-1.5], [-2.0]],
            ),
        },
    )]
