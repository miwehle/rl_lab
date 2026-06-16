from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hpo.lunar_lander import study as study_module
from hpo.lunar_lander.study import neighbors, run_study, select_robust_best


@dataclass
class FakeState:
    name: str


@dataclass
class FakeTrial:
    number: int
    value: float
    params: dict
    user_attrs: dict = field(default_factory=dict)
    state: FakeState = field(default_factory=lambda: FakeState("COMPLETE"))


@dataclass
class FakeStudy:
    trials: list[FakeTrial]
    user_attrs: dict = field(default_factory=dict)

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


def test_neighbors_returns_value_plus_direct_neighbors() -> None:
    assert neighbors(10_000, [2_500, 5_000, 10_000, 20_000]) == [
        5_000,
        10_000,
        20_000,
    ]
    assert neighbors(512, [512, 1024, 2048]) == [512, 1024]


def test_run_study_optimizes_until_target_trial_count(monkeypatch) -> None:
    progress_calls = []
    created_studies = []
    test_dir = Path("fake-hpo-test-dir")

    def fake_create_objective(**_kwargs):
        def objective(trial):
            return float(trial.number)

        return objective

    class FakeOptunaStudy:
        def __init__(self) -> None:
            self.trials = []

        def optimize(self, objective, *, n_trials):
            assert n_trials == 1
            trial = FakeTrial(
                number=len(self.trials),
                value=0.0,
                params={},
            )
            trial.value = objective(trial)
            self.trials.append(trial)

    def fake_create_study(**kwargs):
        assert kwargs["study_name"] == "fake_study"
        assert kwargs["direction"] == "maximize"
        created_studies.append(kwargs)
        return FakeOptunaStudy()

    monkeypatch.setattr(study_module, "create_objective", fake_create_objective)
    monkeypatch.setattr(study_module, "_create_study", fake_create_study)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)

    study = run_study(
        study_name="fake_study",
        search_space=object(),
        n_trials=2,
        num_episodes=3,
        score_window=1,
        output_dir=test_dir / "runs",
        study_dir=test_dir / "studies",
        device="cpu",
        progress_fn=lambda *args, **kwargs: progress_calls.append((args, kwargs)),
    )

    assert len(study.trials) == 2
    assert [trial.value for trial in study.trials] == [0.0, 1.0]
    assert len(progress_calls) == 2
    assert "fake_study.db" in created_studies[0]["storage"]


def test_select_robust_best_rechecks_top_candidates(monkeypatch) -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(number=0, value=100.0, params={"x": 1}),
            FakeTrial(number=1, value=90.0, params={"x": 2}),
            FakeTrial(number=2, value=80.0, params={"x": 3}),
        ]
    )

    def fake_create_objective(**_kwargs):
        def objective(trial):
            trial.set_user_attr("eval_score", float(trial.params["x"] * 10))
            return float(trial.params["x"] * 100)

        return objective

    monkeypatch.setattr(study_module, "create_objective", fake_create_objective)

    params = select_robust_best(
        study=study,
        search_space_factory=lambda: object(),
        num_episodes=3,
        score_window=1,
        device="cpu",
        top_n=2,
        extra_seeds=(1, 2),
    )

    assert params == {"x": 2}
    assert study.user_attrs["robust_best_params"] == {"x": 2}
    assert study.user_attrs["robust_best_objective_score"] == pytest.approx(490 / 3)
    assert study.user_attrs["robust_best_eval_score"] == 20.0


def test_select_robust_best_rejects_empty_study() -> None:
    with pytest.raises(ValueError, match="no complete trials"):
        select_robust_best(
            study=FakeStudy(trials=[]),
            search_space_factory=lambda: object(),
            num_episodes=3,
            score_window=1,
            device="cpu",
        )
