from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hpo import study as study_module
from hpo.evaluation.scoring import ScoringConfig
from hpo.objective import TrialConfig
from hpo.study import StudyRunner, neighbors, run_study, select_robust_best


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


def test_study_runner_reuses_context_and_previous_studies(
    monkeypatch,
) -> None:
    studies = [
        FakeStudy([], {"robust_best_params": {}}),
        FakeStudy([], {"robust_best_params": {"x": 2}}),
    ]
    run_calls = []
    robust_calls = []
    progress_calls = []
    sync_calls = []

    monkeypatch.setattr(
        study_module,
        "run_study",
        lambda **kwargs: run_calls.append(kwargs) or studies[len(run_calls) - 1],
    )
    monkeypatch.setattr(
        study_module,
        "select_robust_best",
        lambda **kwargs: robust_calls.append(kwargs) or {"x": 2},
    )
    monkeypatch.setattr(
        study_module,
        "show_lander_live_progress",
        lambda *args, **kwargs: progress_calls.append((args, kwargs)),
    )

    environment_factory = FakeEnvironmentFactory()
    trial_cfg = TrialConfig(device="cpu")
    runner = StudyRunner(
        storage_path=lambda name: Path("runs") / f"{name}.db",
        environment_factory=environment_factory,
        trial_cfg=trial_cfg,
        study_attrs={"mode": "8d"},
        extra_seeds=(1,),
        sync_fn=lambda: sync_calls.append(None),
    )

    baseline, baseline_params = runner.run(
        "s0",
        search_space="baseline-space",
        n_trials=3,
        scoring_cfg=ScoringConfig(),
        robust=False,
    )
    optimized, best_params = runner.run(
        "s1",
        search_space="search-space",
        n_trials=4,
        scoring_cfg=ScoringConfig(),
    )

    assert baseline_params == {}
    assert best_params == {"x": 2}
    assert runner.studies == [baseline, optimized]
    assert run_calls[1]["storage_path"] == Path("runs/s1.db")
    assert run_calls[1]["environment_factory"] is environment_factory
    assert run_calls[1]["study_attrs"] == {"mode": "8d"}
    assert run_calls[1]["sync_fn"] is runner.sync_fn
    assert robust_calls[0]["search_space"] == "search-space"
    assert robust_calls[0]["extra_seeds"] == (1,)
    assert progress_calls[-1][1]["lander_studies"] == [baseline, optimized]
    assert len(sync_calls) == 1


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

    sync_calls = []
    study = run_study(
        study_name="s0",
        search_space=object(),
        n_trials=2,
        storage_path=Path("runs") / "series.db",
        environment_factory=FakeEnvironmentFactory(),
        study_attrs={"observation_mode": "8d"},
        progress_fn=None,
        sync_fn=lambda: sync_calls.append(None),
    )

    assert "series.db" in created_studies[0]["storage"]
    assert study.user_attrs["observation_mode"] == "8d"
    assert study.user_attrs["baseline_env_steps"] == 10
    assert study.user_attrs["baseline_processed_samples"] == 20
    assert len(sync_calls) == 3


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
        search_space=object(),
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
            search_space=object(),
            environment_factory=FakeEnvironmentFactory(),
            trial_cfg=TrialConfig(),
            scoring_cfg=ScoringConfig(
                baseline_env_steps=10,
                baseline_processed_samples=20,
            ),
        )
