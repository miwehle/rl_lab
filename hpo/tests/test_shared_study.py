from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hpo import study as study_module
from hpo.evaluation.scoring import ScoringConfig
from hpo.objective import TrialConfig
from hpo.study import neighbors, run_study, select_robust_best


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


class FakeEnvironmentFactory:
    pass


def test_neighbors_returns_value_plus_direct_neighbors() -> None:
    assert neighbors(10_000, [2_500, 5_000, 10_000, 20_000]) == [
        5_000,
        10_000,
        20_000,
    ]


def test_run_study_uses_shared_storage_and_task_attrs(monkeypatch) -> None:
    created_studies = []

    def fake_create_objective(**_kwargs):
        def objective(trial):
            trial.user_attrs["env_steps"] = 10
            trial.user_attrs["processed_samples"] = 20
            trial.user_attrs["gym_score"] = 30
            return float(trial.number)

        return objective

    class FakeOptunaStudy:
        def __init__(self) -> None:
            self.trials = []
            self.user_attrs = {}

        def set_user_attr(self, name, value):
            self.user_attrs[name] = value

        def optimize(self, objective, *, n_trials):
            assert n_trials == 1
            trial = FakeTrial(len(self.trials), 0.0, {})
            trial.value = objective(trial)
            self.trials.append(trial)

    def fake_create_study(**kwargs):
        created_studies.append(kwargs)
        return FakeOptunaStudy()

    monkeypatch.setattr(study_module, "create_objective", fake_create_objective)
    monkeypatch.setattr(study_module, "_create_study", fake_create_study)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)

    study = run_study(
        study_name="s0",
        search_space=object(),
        n_trials=2,
        storage_path=Path("runs") / "series.db",
        environment_factory=FakeEnvironmentFactory(),
        study_attrs={"observation_mode": "8d"},
        progress_fn=None,
    )

    assert "series.db" in created_studies[0]["storage"]
    assert study.user_attrs["observation_mode"] == "8d"
    assert study.user_attrs["baseline_env_steps"] == 10
    assert study.user_attrs["baseline_processed_samples"] == 20


def test_select_robust_best_uses_shared_objective(monkeypatch) -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                0,
                100.0,
                {"x": 1},
                {"gym_score": 10.0, "training_effort": 1.0},
            ),
            FakeTrial(
                1,
                90.0,
                {"x": 2},
                {"gym_score": 20.0, "training_effort": 2.0},
            ),
        ],
    )

    def fake_create_objective(**_kwargs):
        def objective(trial):
            trial.set_user_attr("gym_score", float(trial.params["x"] * 10))
            trial.set_user_attr("training_effort", float(trial.params["x"]))
            return float(trial.params["x"] * 100)

        return objective

    monkeypatch.setattr(study_module, "create_objective", fake_create_objective)

    params = select_robust_best(
        study=study,
        search_space_factory=lambda: object(),
        environment_factory=FakeEnvironmentFactory(),
        trial_cfg=TrialConfig(device="cpu"),
        scoring_cfg=ScoringConfig(
            baseline_env_steps=10,
            baseline_processed_samples=20,
        ),
        top_n=2,
        extra_seeds=(1,),
    )

    assert params == {"x": 2}
    assert study.user_attrs["robust_best_objective_score"] == 145
    assert study.user_attrs["robust_best_gym_score"] == 20
    assert study.user_attrs["robust_best_training_effort"] == 2


def test_select_robust_best_rejects_empty_study() -> None:
    with pytest.raises(ValueError, match="no complete trials"):
        select_robust_best(
            study=FakeStudy(trials=[]),
            search_space_factory=lambda: object(),
            environment_factory=FakeEnvironmentFactory(),
            trial_cfg=TrialConfig(),
            scoring_cfg=ScoringConfig(
                baseline_env_steps=10,
                baseline_processed_samples=20,
            ),
        )
