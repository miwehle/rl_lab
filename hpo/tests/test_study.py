from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hpo import study as study_module
from hpo.study import Baseline, StudyRunner, run_study, select_robust_best
from hpo.study_reporting import RobustnessProgress
from common import objective_config


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


@dataclass
class FakeReporter:
    optimization_calls: list = field(default_factory=list)
    robustness_calls: list = field(default_factory=list)

    def report_optimization(self, *args, **kwargs) -> None:
        self.optimization_calls.append((args, kwargs))

    def report_robustness_evaluation(self, *args, **kwargs) -> None:
        self.robustness_calls.append((args, kwargs))


def test_study_runner_reuses_context_and_previous_studies(
    monkeypatch,
) -> None:
    studies = [
        FakeStudy([], {
            "robust_best_params": {"x": 2},
            "robust_best_score": 1.0,
        }),
        FakeStudy([], {
            "robust_best_params": {"x": 3},
            "robust_best_score": 2.0,
        }),
    ]
    run_calls = []
    robust_calls = []
    sync_calls = []

    monkeypatch.setattr(
        study_module,
        "run_study",
        lambda **kwargs: run_calls.append(kwargs) or studies[len(run_calls) - 1],
    )
    monkeypatch.setattr(
        study_module,
        "select_robust_best",
        lambda **kwargs: robust_calls.append(kwargs)
        or studies[len(robust_calls) - 1].user_attrs["robust_best_params"],
    )
    environment_factory = FakeEnvironmentFactory()
    objective_cfg = objective_config(
        environment_factory=environment_factory,
        device="cpu",
    )
    reporter = FakeReporter()
    runner = StudyRunner(
        database_path=lambda name: Path("runs") / f"{name}.db",
        objective_cfg=objective_cfg,
        baseline=Baseline(params={"x": 1}, score=0.0),
        reporter=reporter,
        study_attrs={"mode": "8d"},
        robust_candidates=5,
        extra_seeds=(1,),
        sync_fn=lambda: sync_calls.append(None),
    )

    runner.run(
        "s1",
        suggest_parameter_values="suggest-values",
        n_trials=4,
    )
    runner.run(
        "s2",
        suggest_parameter_values="next-values",
        n_trials=5,
    )

    assert runner.incumbent_params == {"x": 3}
    assert runner.incumbent_score == 2.0
    assert runner.studies == studies
    assert run_calls[1]["database_path"] == Path("runs/s2.db")
    assert run_calls[1]["objective_cfg"].environment_factory is environment_factory
    assert run_calls[1]["study_attrs"] == {"mode": "8d"}
    assert run_calls[1]["sync_fn"] is runner.sync_fn
    assert robust_calls[0]["suggest_parameter_values"] == "suggest-values"
    assert robust_calls[0]["top_n"] == 5
    assert robust_calls[0]["extra_seeds"] == (1,)
    progress = RobustnessProgress(
        candidate_index=1,
        candidate_count=3,
        seed_index=1,
        seed_count=1,
        candidate_seed_scores=[[1.0], [0.5], [0.0]],
    )
    robust_calls[0]["progress_fn"](progress)
    assert reporter.robustness_calls[0][1]["progress"] == progress
    assert reporter.robustness_calls[0][1]["incumbent_params"] == {"x": 3}
    assert reporter.optimization_calls[-1][1]["studies"] == studies
    assert reporter.optimization_calls[-1][1]["incumbent_params"] == {"x": 3}
    assert len(sync_calls) == 2
    assert studies[-1].user_attrs["incumbent_params"] == {"x": 3}
    assert studies[-1].user_attrs["incumbent_score"] == 2.0


def test_study_runner_keeps_better_incumbent(monkeypatch) -> None:
    current = FakeStudy([], {
        "robust_best_params": {"x": 2},
        "robust_best_score": 9.0,
    })
    monkeypatch.setattr(study_module, "run_study", lambda **_kwargs: current)
    monkeypatch.setattr(
        study_module,
        "select_robust_best",
        lambda **_kwargs: {"x": 2},
    )
    runner = StudyRunner(
        database_path=lambda _name: Path("runs/study.db"),
        objective_cfg=objective_config(
            environment_factory=FakeEnvironmentFactory(),
        ),
        baseline=Baseline(params={"x": 1}, score=10.0),
        reporter=FakeReporter(),
    )

    runner.run(
        "s2",
        suggest_parameter_values=object(),
        n_trials=1,
    )

    assert runner.incumbent_params == {"x": 1}
    assert runner.incumbent_score == 10.0
    assert current.user_attrs["robust_best_params"] == {"x": 2}
    assert current.user_attrs["robust_best_score"] == 9.0
    assert current.user_attrs["incumbent_params"] == {"x": 1}
    assert current.user_attrs["incumbent_score"] == 10.0

def test_run_study_uses_shared_storage_and_task_attrs(monkeypatch) -> None:
    created_studies = []

    def fake_create_objective(**_kwargs):
        def objective(trial):
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
    progress_trial_counts = []
    study = run_study(
        study_name="s1",
        suggest_parameter_values=object(),
        incumbent_params={},
        n_trials=2,
        database_path=Path("runs") / "series.db",
        objective_cfg=objective_config(
            environment_factory=FakeEnvironmentFactory(),
        ),
        study_attrs={"observation_mode": "8d"},
        progress_fn=lambda current_study, **_kwargs: progress_trial_counts.append(
            len(current_study.trials)
        ),
        sync_fn=lambda: sync_calls.append(None),
    )

    assert "series.db" in created_studies[0]["storage"]
    assert study.user_attrs["observation_mode"] == "8d"
    assert study.user_attrs["eval_episodes"] == 20
    assert len(sync_calls) == 2
    assert progress_trial_counts == [0, 1, 2]


def test_select_robust_best_uses_shared_objective(monkeypatch) -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(
                0,
                100.0,
                {"x": 1},
            ),
            FakeTrial(
                1,
                90.0,
                {"x": 2},
            ),
        ],
    )

    fixed_trials = []

    def fake_create_objective(**_kwargs):
        def objective(trial):
            fixed_trials.append(trial)
            return float(trial.params["x"] * 100)

        return objective

    monkeypatch.setattr(study_module, "create_objective", fake_create_objective)
    progress_calls = []

    def record_progress(progress):
        progress_calls.append(RobustnessProgress(
            candidate_index=progress.candidate_index,
            candidate_count=progress.candidate_count,
            seed_index=progress.seed_index,
            seed_count=progress.seed_count,
            candidate_seed_scores=[
                list(scores) for scores in progress.candidate_seed_scores
            ],
        ))

    params = select_robust_best(
        study=study,
        suggest_parameter_values=object(),
        incumbent_params={},
        objective_cfg=objective_config(
            environment_factory=FakeEnvironmentFactory(),
            device="cpu",
        ),
        top_n=2,
        extra_seeds=(1,),
        progress_fn=record_progress,
    )

    assert params == {"x": 2}
    assert study.user_attrs["robust_best_score"] == 145
    assert [
        (call.candidate_index, call.seed_index)
        for call in progress_calls
    ] == [(1, 1), (1, 1), (2, 1), (2, 1)]
    assert progress_calls[-1].candidate_seed_scores == [
        [100.0, 100.0],
        [90.0, 200.0],
    ]
    assert [
        (trial.number, trial.checkpoint_subdir, trial.checkpoint_stem)
        for trial in fixed_trials
    ] == [
        (0, "robustness", "trial_0000_seed_1"),
        (1, "robustness", "trial_0001_seed_1"),
    ]


def test_select_robust_best_rejects_empty_study() -> None:
    with pytest.raises(ValueError, match="no complete trials"):
        select_robust_best(
            study=FakeStudy(trials=[]),
            suggest_parameter_values=object(),
            incumbent_params={},
            objective_cfg=objective_config(
                environment_factory=FakeEnvironmentFactory(),
            ),
        )


def test_baseline_loads_incumbent_from_database(monkeypatch) -> None:
    study = FakeStudy([], {
        "incumbent_params": {"x": 2},
        "incumbent_score": 123.0,
    })
    load_calls = []
    monkeypatch.setattr(
        study_module,
        "_load_study",
        lambda **kwargs: load_calls.append(kwargs) or study,
    )

    baseline = Baseline.from_database(Path("runs/previous.db"), "s4")

    assert baseline == Baseline(params={"x": 2}, score=123.0)
    assert load_calls[0]["study_name"] == "s4"
    assert "previous.db" in load_calls[0]["storage"]
