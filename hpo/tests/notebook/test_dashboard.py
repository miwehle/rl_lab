from dataclasses import dataclass, field

import pytest

from hpo.notebook import dashboard
from hpo.notebook.dashboard import Dashboard
from hpo.notebook.dashboard import main as dashboard_main
from hpo.study_reporting import RobustnessProgress, TrainingProgress


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


def test_dashboard_rejects_unknown_render_mode() -> None:
    with pytest.raises(ValueError, match="unsupported dashboard render_mode"):
        Dashboard(render_mode="smooth")


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
        dashboard_main,
        "build_dashboard",
        lambda **kwargs: ("dashboard", kwargs),
    )
    monkeypatch.setattr(dashboard_main, "_display", displayed.append)
    monkeypatch.setattr(
        dashboard_main, "_clear_output", lambda **kwargs: cleared.append(kwargs)
    )
    reporter = Dashboard()
    reporter.set_study_series_context(
        studies=[study],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
    )

    reporter.report_optimization(
        study,
        target_trials=40,
    )

    assert cleared == [{"wait": True}]
    assert displayed[-1] == (
        "dashboard",
        {
            "study": study,
            "target_trials": 40,
            "studies": [study],
            "incumbent_params": {
                "learning_rate": 0.001,
                "gamma": 0.99,
                },
                "robustness_progress": None,
                "training_progress": None,
                "training_score_min": -500.0,
            },
        )


def test_show_dashboard_during_training_reuses_current_context(monkeypatch) -> None:
    study = FakeStudy(trials=[], user_attrs={"incumbent_score": 180})
    displayed = []
    monkeypatch.setattr(
        dashboard_main,
        "build_dashboard",
        lambda **kwargs: ("dashboard", kwargs),
    )
    monkeypatch.setattr(dashboard_main, "_display", displayed.append)
    monkeypatch.setattr(dashboard_main, "_clear_output", lambda **_kwargs: None)
    reporter = Dashboard()
    reporter.set_study_series_context(
        studies=[study],
        incumbent_params={"learning_rate": 0.001},
    )
    reporter.report_optimization(
        study,
        target_trials=40,
    )
    progress = TrainingProgress(
        trial_number=3,
        target_episodes=5,
        episode_returns=[1.0],
    )

    reporter.report_training_progress(progress)

    assert displayed[-1][1]["study"] is study
    assert displayed[-1][1]["target_trials"] == 40
    assert displayed[-1][1]["training_progress"] == progress


def test_dashboard_throttles_training_updates_but_shows_final(monkeypatch) -> None:
    study = FakeStudy(trials=[], user_attrs={"incumbent_score": 180})
    displayed = []
    times = iter([10.0, 11.0, 12.0])
    monkeypatch.setattr(dashboard_main, "perf_counter", lambda: next(times))
    monkeypatch.setattr(dashboard_main, "build_dashboard", lambda **kwargs: kwargs)
    monkeypatch.setattr(dashboard_main, "_display", displayed.append)
    monkeypatch.setattr(dashboard_main, "_clear_output", lambda **_kwargs: None)
    reporter = Dashboard(training_update_interval_seconds=5.0)
    reporter.set_study_series_context(studies=[study], incumbent_params={})
    reporter.report_optimization(study, target_trials=40)

    reporter.report_training_progress(TrainingProgress(1, 3, [1.0]))
    reporter.report_training_progress(TrainingProgress(1, 3, [1.0, 2.0]))
    reporter.report_training_progress(TrainingProgress(1, 3, [1.0, 2.0, 3.0]))

    training_updates = [
        item["training_progress"] for item in displayed if item["training_progress"]
    ]
    assert len(training_updates) == 2
    assert training_updates[-1].episode_returns == [1.0, 2.0, 3.0]


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
    assert figure.layout.height == 850
    assert [annotation.text for annotation in figure.layout.annotations[:4]] == [
        "Study Series",
        "Current HPs",
        "Study: S1 Update Economy",
        "Checkpoint Robustness",
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
    assert figure.layout.legend.y < 0
    assert figure.layout.legend.yanchor == "top"
    assert figure.layout.margin.b >= 85
    no_data_annotations = [
        annotation
        for annotation in figure.layout.annotations
        if annotation.text == "No data yet"
    ]
    assert len(no_data_annotations) == 2
    assert figure.layout.xaxis3.showticklabels is False
    assert figure.layout.yaxis3.showticklabels is False
    assert figure.layout.xaxis3.showgrid is False
    assert figure.layout.yaxis3.showgrid is False
    assert figure.layout.xaxis4.showticklabels is False
    assert figure.layout.yaxis4.showticklabels is False
    assert figure.layout.yaxis5.showticklabels is False
    assert figure.layout.xaxis4.showgrid is False
    assert figure.layout.yaxis4.showgrid is False
    assert figure.layout.yaxis5.showgrid is False
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
    } == set()


def test_study_plot_uses_evaluation_checkpoint_score() -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                number=0,
                value=50.0,
                user_attrs={"evaluation_checkpoint_score": 120.0},
            ),
            FakeTrial(
                number=1,
                value=80.0,
                user_attrs={},
            ),
        ],
    )

    figure = dashboard.build_dashboard(
        study=study,
        target_trials=2,
        studies=[],
        incumbent_params={},
    )

    score_trace = next(
        trace
        for trace in figure.data
        if trace.name == "Score" and trace.xaxis == "x2"
    )
    best_trace = next(
        trace
        for trace in figure.data
        if trace.name == "Best score" and trace.xaxis == "x2"
    )

    assert list(score_trace.y) == [120.0, 80.0]
    assert list(best_trace.y) == [120.0, 120.0]


def test_current_hps_use_live_trial_params_during_training() -> None:
    study = FakeStudy(trials=[], study_name="s1_update_economy")

    figure = dashboard.build_dashboard(
        study=study,
        target_trials=40,
        studies=[],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
        training_progress=TrainingProgress(
            trial_number=3,
            target_episodes=5,
            episode_returns=[1.0],
            trial_params={
                "learning_rate": 0.0022727854024196057,
                "eps_end": 0.047716002108220544,
                "gamma": 0.99,
            },
            optimized_param_names=["learning_rate", "eps_end"],
        ),
    )

    table = next(trace for trace in figure.data if trace.type == "table")
    assert list(table.cells.values[0]) == ["learning_rate", "eps_end", "gamma"]
    assert list(table.cells.values[1]) == ["0.002273", "0.04772", "0.99"]
    assert list(table.cells.fill.color[0]) == ["#fff2cc", "#fff2cc", "white"]


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


def test_checkpoint_robustness_plot_shows_candidate_intervals() -> None:
    study = FakeStudy(trials=[], study_name="s1_update_economy")
    figure = dashboard.build_dashboard(
        study=study,
        target_trials=40,
        studies=[],
        incumbent_params={},
        robustness_progress=RobustnessProgress(
            candidate_index=2,
            candidate_count=3,
            seed_index=1,
            seed_count=1,
            candidate_seed_scores=[[200.0, 240.0], [210.0, 250.0], [190.0]],
            title="Checkpoint Robustness Evaluation",
            step_label="Eval",
            first_score_label="Source score",
            extra_score_label="Robust eval",
            checkpoint_summaries=[
                {
                    "candidate": 1,
                    "trial_number": 35,
                    "mean": 240.0,
                    "median": 245.0,
                    "min": 100.0,
                    "q05": 180.0,
                    "q25": 220.0,
                    "q75": 260.0,
                    "q95": 300.0,
                    "max": 320.0,
                },
                {
                    "candidate": 2,
                    "trial_number": 42,
                    "mean": 250.0,
                    "median": 248.0,
                    "min": 150.0,
                    "q05": 200.0,
                    "q25": 230.0,
                    "q75": 270.0,
                    "q95": 310.0,
                    "max": 330.0,
                },
            ],
        ),
    )

    min_max = [trace for trace in figure.data if trace.name == "min..max"]
    mean = next(trace for trace in figure.data if trace.name == "mean")

    assert list(min_max[0].x) == [100.0, 320.0]
    assert list(min_max[0].y) == ["C1 trial 35", "C1 trial 35"]
    assert list(mean.x) == [240.0, 250.0]
    assert list(mean.y) == ["C1 trial 35", "C2 trial 42"]
    assert mean.marker.color == "white"
    assert mean.marker.line.color == "black"
    assert not any(trace.name == "median" for trace in figure.data)
    assert figure.layout.xaxis3.title.text == "Gym score"
    assert figure.layout.yaxis3.title.text == "Checkpoint"
    assert {
        trace.name for trace in figure.data
        if getattr(trace, "showlegend", False) is not False
        and trace.name is not None
    } == set()
    assert figure.layout.annotations[3].text.startswith(
        "Checkpoint Robustness Evaluation"
    )


def test_dashboard_shows_stored_checkpoint_robustness_after_study() -> None:
    study = FakeStudy(
        trials=[],
        study_name="s1_update_economy",
        user_attrs={
            "checkpoint_robustness": [
                {
                    "trial_number": 3,
                    "checkpoint_summary": {
                        "candidate": 1,
                        "trial_number": 3,
                        "mean": 145.4,
                        "median": 150.0,
                        "min": 80.0,
                        "q05": 90.0,
                        "q25": 120.0,
                        "q75": 170.0,
                        "q95": 210.0,
                        "max": 220.0,
                    },
                }
            ],
        },
    )

    figure = dashboard.build_dashboard(
        study=study,
        target_trials=10,
        studies=[study],
        incumbent_params={},
    )

    mean = next(trace for trace in figure.data if trace.name == "mean")

    assert list(mean.x) == [145.4]
    assert list(mean.y) == ["C1 trial 3"]
    assert figure.layout.xaxis3.title.text == "Gym score"
    assert not any(
        annotation.text == "Waiting for robustness evaluation"
        for annotation in figure.layout.annotations
    )


def test_empty_stored_checkpoint_robustness_hides_panel_axes() -> None:
    study = FakeStudy(
        trials=[],
        study_name="s1_update_economy",
        user_attrs={"checkpoint_robustness": []},
    )

    figure = dashboard.build_dashboard(
        study=study,
        target_trials=10,
        studies=[study],
        incumbent_params={},
    )

    assert any(annotation.text == "No data yet" for annotation in figure.layout.annotations)
    assert figure.layout.xaxis3.showticklabels is False
    assert figure.layout.yaxis3.showticklabels is False
    assert figure.layout.xaxis3.showgrid is False
    assert figure.layout.yaxis3.showgrid is False


def test_training_plot_shows_returns_trailing_mean_and_checkpoint_reference() -> None:
    study = FakeStudy(trials=[], study_name="s1_update_economy")
    figure = dashboard.build_dashboard(
        study=study,
        target_trials=40,
        studies=[],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
        training_progress=TrainingProgress(
            trial_number=3,
            target_episodes=5,
            episode_returns=[1.0, 3.0, 5.0, 7.0],
            episode_epsilons=[0.9, 0.8, 0.7, 0.6],
            episode_env_labels=["moon", "mars", "mercury", "earth"],
            checkpoint_window=2,
            checkpoint_min_score=2.5,
            best_checkpoint_score=4.0,
        ),
    )

    returns = next(trace for trace in figure.data if trace.name == "Episode return")
    trailing_mean = next(
        trace for trace in figure.data if trace.name == "Mean (2 episodes)"
    )
    checkpoint = next(
        trace for trace in figure.data if trace.name == "Best checkpoint score"
    )
    epsilon = next(trace for trace in figure.data if trace.name == "Epsilon")
    moon = next(trace for trace in figure.data if trace.name == "moon")
    mars = next(trace for trace in figure.data if trace.name == "mars")
    env_traces = [
        trace.name
        for trace in figure.data
        if trace.name in {"mercury", "earth", "moon", "mars"}
    ]

    assert list(returns.x) == [1, 2, 3, 4]
    assert list(returns.y) == [1.0, 3.0, 5.0, 7.0]
    assert returns.mode == "lines"
    assert returns.line.color == "#9a9a9a"
    assert env_traces == ["mercury", "earth", "moon", "mars"]
    assert {
        trace.name for trace in figure.data
        if getattr(trace, "showlegend", False) is not False
        and trace.name is not None
    } == {"mercury", "earth", "moon", "mars"}
    assert list(moon.x) == [1]
    assert list(moon.y) == [1.0]
    assert list(mars.x) == [2]
    assert list(mars.y) == [3.0]
    assert list(epsilon.x) == [1, 2, 3, 4]
    assert list(epsilon.y) == [0.9, 0.8, 0.7, 0.6]
    assert list(trailing_mean.x) == [2, 3, 4]
    assert list(trailing_mean.y) == pytest.approx([2.0, 4.0, 6.0])
    assert list(checkpoint.y) == [4.0, 4.0]
    assert figure.layout.xaxis4.title.text == "Episode"
    assert figure.layout.yaxis4.title.text == "Gym score"
    assert figure.layout.yaxis5.title.text == "Epsilon"
    assert any(
        annotation.text
        == "Current Trial Training - Trial 3 - Mean (2 episodes): 6.0 - Best Mean: 6.0"
        for annotation in figure.layout.annotations
    )


def test_training_plot_starts_reference_at_checkpoint_threshold() -> None:
    study = FakeStudy(trials=[], study_name="s1_update_economy")
    figure = dashboard.build_dashboard(
        study=study,
        target_trials=40,
        studies=[],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
        training_progress=TrainingProgress(
            trial_number=3,
            target_episodes=5,
            episode_returns=[1.0],
            checkpoint_window=2,
            checkpoint_min_score=10.0,
        ),
    )

    threshold = next(
        trace for trace in figure.data if trace.name == "Checkpoint threshold"
    )

    assert list(threshold.y) == [10.0, 10.0]


def test_training_plot_clips_lower_score_axis_by_default() -> None:
    study = FakeStudy(trials=[], study_name="s1_update_economy")
    progress = TrainingProgress(
        trial_number=3,
        target_episodes=5,
        episode_returns=[-3000.0, 10.0],
    )

    clipped = dashboard.build_dashboard(
        study=study,
        target_trials=40,
        studies=[],
        incumbent_params={},
        training_progress=progress,
    )
    unclipped = dashboard.build_dashboard(
        study=study,
        target_trials=40,
        studies=[],
        incumbent_params={},
        training_progress=progress,
        training_score_min=None,
    )

    assert list(clipped.layout.yaxis4.range) == [-500, 20.0]
    assert list(unclipped.layout.yaxis4.range) == [-3010.0, 20.0]


def test_show_dashboard_during_robustness_evaluation_replaces_oh(monkeypatch) -> None:
    study = FakeStudy(trials=[], study_name="s1_update_economy")
    displayed = []
    cleared = []
    monkeypatch.setattr(
        dashboard_main,
        "build_dashboard",
        lambda **kwargs: ("dashboard", kwargs),
    )
    monkeypatch.setattr(dashboard_main, "_display", displayed.append)
    monkeypatch.setattr(
        dashboard_main, "_clear_output", lambda **kwargs: cleared.append(kwargs)
    )
    reporter = Dashboard()
    reporter.set_study_series_context(
        studies=[],
        incumbent_params={"learning_rate": 0.001, "gamma": 0.99},
    )
    reporter.report_optimization(study, target_trials=0)

    reporter.report_robustness_evaluation(
        progress=RobustnessProgress(
            candidate_index=2,
            candidate_count=3,
            seed_index=1,
            seed_count=1,
            candidate_seed_scores=[[-1.0], [-1.5], [-2.0]],
        ),
    )

    assert cleared == [{"wait": True}, {"wait": True}]
    assert displayed[-1] == (
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
                "training_progress": None,
                "training_score_min": -500.0,
            },
        )
