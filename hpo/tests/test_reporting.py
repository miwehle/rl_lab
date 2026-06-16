from dataclasses import dataclass, field

from hpo.evaluation.reporting import plot_lander_progress


@dataclass
class FakeState:
    name: str


@dataclass
class FakeTrial:
    number: int
    user_attrs: dict
    state: FakeState = field(default_factory=lambda: FakeState("COMPLETE"))


@dataclass
class FakeStudy:
    trials: list[FakeTrial]


def test_plot_lander_progress_uses_cumulative_training_time_and_eval_score() -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                number=1,
                user_attrs={"wall_time_seconds": 120, "eval_score": 210},
            ),
            FakeTrial(
                number=0,
                user_attrs={"wall_time_seconds": 60, "eval_score": 180},
            ),
            FakeTrial(
                number=2,
                user_attrs={"wall_time_seconds": 999, "other": 1},
            ),
        ]
    )

    fig = plot_lander_progress(study)
    ax = fig.axes[0]

    assert list(ax.lines[0].get_xdata()) == [1.0, 3.0]
    assert list(ax.lines[0].get_ydata()) == [180.0, 210.0]
    assert ax.get_xlabel() == "Cumulative L4 training time (min)"
    assert ax.get_ylabel() == "Greedy eval score"
    assert [line.get_ydata()[0] for line in ax.lines[1:]] == [200, 250]
