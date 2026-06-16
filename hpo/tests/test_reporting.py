from dataclasses import dataclass, field

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


def test_plot_lander_progress_uses_mean_training_time_and_best_eval_score() -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                number=1,
                value=10,
                user_attrs={"wall_time_seconds": 120, "eval_score": 210},
            ),
            FakeTrial(
                number=0,
                value=20,
                user_attrs={"wall_time_seconds": 60, "eval_score": 180},
            ),
            FakeTrial(
                number=2,
                value=30,
                user_attrs={"wall_time_seconds": 999, "other": 1},
            ),
        ],
        study_name="s1_update_economy",
    )

    fig = plot_lander_progress(study)
    ax = fig.axes[0]

    assert list(ax.lines[0].get_xdata()) == [6.55]
    assert list(ax.lines[0].get_ydata()) == [180.0]
    assert ax.get_xlabel() == "Mean L4 training time per Lander (min)"
    assert ax.get_ylabel() == "Greedy eval score"
    assert [line.get_ydata()[0] for line in ax.lines[1:]] == [200, 250]


def test_plot_lander_progress_uses_one_point_per_study() -> None:
    studies = [
        FakeStudy(
            trials=[
                FakeTrial(
                    number=0,
                    value=1,
                    user_attrs={"wall_time_seconds": 60, "eval_score": 180},
                ),
            ],
            study_name="s0_baseline",
        ),
        FakeStudy(
            trials=[
                FakeTrial(
                    number=0,
                    value=1,
                    user_attrs={"wall_time_seconds": 120, "eval_score": 210},
                ),
            ],
            study_name="s1_update_economy",
        ),
    ]

    fig = plot_lander_progress(studies)
    ax = fig.axes[0]

    assert list(ax.lines[0].get_xdata()) == [1.0, 2.0]
    assert list(ax.lines[0].get_ydata()) == [180.0, 210.0]


def test_plot_lander_progress_prefers_robust_eval_score() -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                number=0,
                value=1,
                user_attrs={"wall_time_seconds": 60, "eval_score": 180},
            ),
        ],
        user_attrs={"robust_best_eval_score": 215},
    )

    fig = plot_lander_progress(study)
    ax = fig.axes[0]

    assert list(ax.lines[0].get_ydata()) == [215.0]


def test_show_lander_live_progress_displays_lander_and_optuna_history() -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                number=0,
                value=1,
                user_attrs={"wall_time_seconds": 60, "eval_score": 180},
            ),
        ]
    )
    displayed = []
    clear_calls = []

    show_lander_live_progress(
        study,
        target_trials=40,
        lander_studies=[study],
        clear_output_fn=lambda **kwargs: clear_calls.append(kwargs),
        display_fn=displayed.append,
        plot_history=lambda current_study: ("oh", current_study),
    )

    assert clear_calls == [{"wait": True}]
    assert len(displayed) == 2
    assert displayed[0].axes[0].get_ylabel() == "Greedy eval score"
    assert displayed[1] == ("oh", study)
