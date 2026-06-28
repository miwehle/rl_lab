from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hpo import study as study_module
from hpo.checkpointing import ObjectiveHookFactory
from hpo.study import Incumbent, StudyRunner, run_study
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


@dataclass
class FakeStudySummary:
    study_name: str
    user_attrs: dict = field(default_factory=dict)


class FakeEnvironmentFactory:
    pass


@dataclass
class FakeReporter:
    context_calls: list = field(default_factory=list)
    optimization_calls: list = field(default_factory=list)
    robustness_calls: list = field(default_factory=list)
    training_calls: list = field(default_factory=list)

    def set_study_series_context(self, *args, **kwargs) -> None:
        self.context_calls.append((args, kwargs))

    def report_optimization(self, *args, **kwargs) -> None:
        self.optimization_calls.append((args, kwargs))

    def report_robustness_evaluation(self, *args, **kwargs) -> None:
        self.robustness_calls.append((args, kwargs))

    def report_training_progress(self, *args, **kwargs) -> None:
        self.training_calls.append((args, kwargs))


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
        "_create_or_load_study",
        lambda **kwargs: studies[len(run_calls)],
    )
    monkeypatch.setattr(
        study_module,
        "run_study",
        lambda **kwargs: run_calls.append(kwargs) or kwargs["study"],
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
        baseline=Incumbent(params={"x": 1}, score=0.0),
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
    assert run_calls[1]["study"] is studies[1]
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
    assert reporter.robustness_calls[0][0] == (progress,)
    assert reporter.context_calls[-1][1]["studies"] == studies
    assert reporter.context_calls[-1][1]["incumbent_params"] == {"x": 3}
    assert len(sync_calls) == 2
    assert studies[-1].user_attrs["incumbent_params"] == {"x": 3}
    assert studies[-1].user_attrs["incumbent_score"] == 2.0


def test_study_runner_loads_finished_study_series_for_dashboard(
    monkeypatch,
    tmp_path,
) -> None:
    database_path = tmp_path / "series.db"
    database_path.touch()
    previous = FakeStudy([], {"incumbent_score": 5.0})
    current = FakeStudy([], {
        "robust_best_params": {"x": 2},
        "robust_best_score": 9.0,
    })
    monkeypatch.setattr(
        study_module,
        "_all_study_summaries",
        lambda **_kwargs: [
            FakeStudySummary("s2", {"incumbent_score": 5.0}),
            FakeStudySummary("s3", {}),
        ],
    )
    monkeypatch.setattr(study_module, "_load_study", lambda **_kwargs: previous)
    monkeypatch.setattr(
        study_module,
        "_create_or_load_study",
        lambda **_kwargs: current,
    )
    monkeypatch.setattr(study_module, "run_study", lambda **kwargs: kwargs["study"])
    monkeypatch.setattr(study_module, "select_robust_best", lambda **_kwargs: {"x": 2})
    reporter = FakeReporter()
    runner = StudyRunner(
        database_path=lambda _name: database_path,
        objective_cfg=objective_config(environment_factory=FakeEnvironmentFactory()),
        reporter=reporter,
    )

    runner.run("s3", suggest_parameter_values=object(), n_trials=1)

    assert reporter.context_calls[0][1]["studies"] == [previous, current]
    assert runner.studies == [previous, current]


def test_study_runner_keeps_better_incumbent(monkeypatch) -> None:
    current = FakeStudy([], {
        "robust_best_params": {"x": 2},
        "robust_best_score": 9.0,
    })
    monkeypatch.setattr(study_module, "_create_or_load_study", lambda **_kwargs: current)
    monkeypatch.setattr(study_module, "run_study", lambda **kwargs: kwargs["study"])
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
        baseline=Incumbent(params={"x": 1}, score=10.0),
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


def test_study_runner_can_start_without_baseline(monkeypatch, tmp_path) -> None:
    current = FakeStudy([], {
        "robust_best_params": {"x": 2},
        "robust_best_score": 9.0,
    })
    run_calls = []
    robust_calls = []
    monkeypatch.setattr(study_module, "_create_or_load_study", lambda **_kwargs: current)
    monkeypatch.setattr(
        study_module,
        "run_study",
        lambda **kwargs: run_calls.append(
            kwargs | {"incumbent_params": dict(kwargs["incumbent_params"])}
        ) or kwargs["study"],
    )
    monkeypatch.setattr(
        study_module,
        "select_robust_best",
        lambda **kwargs: robust_calls.append(
            kwargs | {"incumbent_params": dict(kwargs["incumbent_params"])}
        ) or {"x": 2},
    )
    runner = StudyRunner(
        database_path=lambda _name: Path("runs/study.db"),
        objective_cfg=objective_config(
            environment_factory=FakeEnvironmentFactory(),
            hooks=ObjectiveHookFactory(tmp_path, window=2),
        ),
        reporter=FakeReporter(),
    )

    runner.run("s1", suggest_parameter_values=object(), n_trials=1)

    assert run_calls[0]["incumbent_params"] == {}
    assert run_calls[0]["objective_cfg"].hooks.min_score is None
    assert robust_calls[0]["incumbent_params"] == {}
    assert robust_calls[0]["objective_cfg"].hooks.min_score is None
    assert runner.incumbent_params == {"x": 2}
    assert runner.incumbent_score == 9.0
    assert current.user_attrs["incumbent_params"] == {"x": 2}
    assert current.user_attrs["incumbent_score"] == 9.0


def test_study_runner_skips_already_finished_study(monkeypatch, capsys) -> None:
    current = FakeStudy(
        [FakeTrial(0, 9.0, {})],
        {"robust_best_score": 9.0, "incumbent_score": 9.0},
    )
    monkeypatch.setattr(study_module, "_create_or_load_study", lambda **_kwargs: current)
    monkeypatch.setattr(
        study_module,
        "run_study",
        lambda **_kwargs: pytest.fail("finished study should not run"),
    )
    monkeypatch.setattr(
        study_module,
        "select_robust_best",
        lambda **_kwargs: pytest.fail("finished study should not select again"),
    )
    reporter = FakeReporter()
    runner = StudyRunner(
        database_path=lambda _name: Path("runs/study.db"),
        objective_cfg=objective_config(
            environment_factory=FakeEnvironmentFactory(),
        ),
        baseline=Incumbent(params={"x": 1}, score=0.0),
        reporter=reporter,
    )

    runner.run("s1", suggest_parameter_values=object(), n_trials=1)

    assert capsys.readouterr().out == "Study already finished.\n"
    assert reporter.context_calls == []
    assert runner.studies == []


def test_study_runner_uses_incumbent_as_checkpoint_min_score(
    monkeypatch,
    tmp_path,
) -> None:
    current = FakeStudy([], {
        "robust_best_params": {"x": 2},
        "robust_best_score": 9.0,
    })
    run_calls = []
    robust_calls = []
    cfg = objective_config(
        environment_factory=FakeEnvironmentFactory(),
        hooks=ObjectiveHookFactory(tmp_path, window=2),
    )
    monkeypatch.setattr(
        study_module,
        "_create_or_load_study",
        lambda **_kwargs: current,
    )
    monkeypatch.setattr(
        study_module,
        "run_study",
        lambda **kwargs: run_calls.append(kwargs) or kwargs["study"],
    )
    monkeypatch.setattr(
        study_module,
        "select_robust_best",
        lambda **kwargs: robust_calls.append(kwargs) or {"x": 2},
    )
    runner = StudyRunner(
        database_path=lambda _name: Path("runs/study.db"),
        objective_cfg=cfg,
        baseline=Incumbent(params={"x": 1}, score=10.0),
        reporter=FakeReporter(),
    )

    runner.run(
        "s2",
        suggest_parameter_values=object(),
        n_trials=1,
    )

    assert run_calls[0]["objective_cfg"].hooks.min_score == pytest.approx(10.0)
    assert robust_calls[0]["objective_cfg"].hooks.min_score == pytest.approx(10.0)
    assert run_calls[0]["objective_cfg"].hooks.training_progress_fn == (
        runner.reporter.report_training_progress
    )
    assert robust_calls[0]["objective_cfg"].hooks.training_progress_fn == (
        runner.reporter.report_training_progress
    )
    assert cfg.hooks.min_score is None


def test_run_study_uses_task_attrs_and_reports_progress(monkeypatch) -> None:
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

    monkeypatch.setattr(study_module, "create_objective", fake_create_objective)

    sync_calls = []
    progress_trial_counts = []
    study = FakeOptunaStudy()
    study = run_study(
        study=study,
        suggest_parameter_values=object(),
        incumbent_params={},
        n_trials=2,
        objective_cfg=objective_config(
            environment_factory=FakeEnvironmentFactory(),
        ),
        study_attrs={"observation_mode": "8d"},
        progress_fn=lambda current_study, **_kwargs: progress_trial_counts.append(
            len(current_study.trials)
        ),
        sync_fn=lambda: sync_calls.append(None),
    )

    assert study.user_attrs["observation_mode"] == "8d"
    assert study.user_attrs["eval_episodes"] == 20
    assert len(sync_calls) == 2
    assert progress_trial_counts == [0, 1, 2]


def test_incumbent_loads_from_study() -> None:
    study = FakeStudy([], {
        "incumbent_params": {"x": 2},
        "incumbent_score": 123.0,
    })

    incumbent = Incumbent.from_study(study)

    assert incumbent == Incumbent(params={"x": 2}, score=123.0)


def test_incumbent_loads_from_database(monkeypatch) -> None:
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

    baseline = Incumbent.from_database(Path("runs/previous.db"), "s4")

    assert baseline == Incumbent(params={"x": 2}, score=123.0)
    assert load_calls[0]["study_name"] == "s4"
    assert "previous.db" in load_calls[0]["storage"]
