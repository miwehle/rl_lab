from dataclasses import dataclass, field

import pytest

from hpo.checkpointing import ObjectiveHookFactory
from hpo.evaluation import hp_robustness as hp_robustness_module
from hpo.evaluation.hp_robustness import select_robust_best
from hpo.study.infra_cfg import InfraCfg
from hpo.study.reporting import RobustnessProgress, TrainingProgress
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


def test_select_robust_best_uses_shared_objective(monkeypatch) -> None:
    study = FakeStudy(trials=[FakeTrial(0, 100.0, {"x": 1}), FakeTrial(1, 90.0, {"x": 2})])

    fixed_trials = []

    def fake_create_objective(**_kwargs):
        def objective(trial):
            fixed_trials.append(trial)
            trial.set_user_attr(
                "checkpoint_path", f"{trial.checkpoint_subdir}/{trial.checkpoint_stem}_best.pt"
            )
            trial.set_user_attr("checkpoint_score", trial.params["x"] * 10)
            return float(trial.params["x"] * 100)

        return objective

    monkeypatch.setattr(hp_robustness_module, "create_objective", fake_create_objective)
    progress_calls = []

    def record_progress(progress):
        progress_calls.append(
            RobustnessProgress(
                candidate_index=progress.candidate_index,
                candidate_count=progress.candidate_count,
                seed_index=progress.seed_index,
                seed_count=progress.seed_count,
                candidate_seed_scores=[list(scores) for scores in progress.candidate_seed_scores],
            )
        )

    params = select_robust_best(
        study=study,
        suggest_parameter_values=object(),
        incumbent_params={},
        objective_cfg=objective_config(environment_factory=FakeEnvironmentFactory(), device="cpu"),
        top_n=2,
        extra_seeds=(1,),
        progress_fn=record_progress,
    )

    assert params == {"x": 2}
    assert study.user_attrs["robust_best_score"] == 145
    assert [(call.candidate_index, call.seed_index) for call in progress_calls] == [
        (1, 1),
        (1, 1),
        (2, 1),
        (2, 1),
    ]
    assert progress_calls[-1].candidate_seed_scores == [[100.0, 100.0], [90.0, 200.0]]
    assert study.user_attrs["robustness_checkpoints"] == [
        {
            "trial_number": 0,
            "seed_offset": 1,
            "score": 100.0,
            "checkpoint_path": "robustness/trial_0000_seed_1_best.pt",
            "checkpoint_score": 10,
        },
        {
            "trial_number": 1,
            "seed_offset": 1,
            "score": 200.0,
            "checkpoint_path": "robustness/trial_0001_seed_1_best.pt",
            "checkpoint_score": 20,
        },
    ]
    assert [(trial.number, trial.checkpoint_subdir, trial.checkpoint_stem) for trial in fixed_trials] == [
        (0, "robustness", "trial_0000_seed_1"),
        (1, "robustness", "trial_0001_seed_1"),
    ]


def test_select_robust_best_ranks_by_evaluation_checkpoint_score(monkeypatch) -> None:
    study = FakeStudy(
        trials=[
            FakeTrial(0, 200.0, {"x": 1}, user_attrs={"evaluation_checkpoint_score": 50.0}),
            FakeTrial(1, 100.0, {"x": 2}, user_attrs={"evaluation_checkpoint_score": 180.0}),
        ]
    )
    fixed_trials = []

    def fake_create_objective(**_kwargs):
        def objective(trial):
            fixed_trials.append(trial)
            return 0.0

        return objective

    monkeypatch.setattr(hp_robustness_module, "create_objective", fake_create_objective)

    select_robust_best(
        study=study,
        suggest_parameter_values=object(),
        incumbent_params={},
        objective_cfg=objective_config(environment_factory=FakeEnvironmentFactory()),
        top_n=1,
        extra_seeds=(1,),
    )

    assert fixed_trials[0].number == 1


def test_select_robust_best_reports_training_progress(monkeypatch, tmp_path) -> None:
    study = FakeStudy(trials=[FakeTrial(0, 100.0, {"x": 1})])
    training_calls = []
    progress_calls = []

    def fake_create_objective(**kwargs):
        def objective(_trial):
            kwargs["config"].hooks.training_progress_fn(
                TrainingProgress(trial_number=0, target_episodes=10, episode_returns=[1.0])
            )
            return 110.0

        return objective

    monkeypatch.setattr(hp_robustness_module, "create_objective", fake_create_objective)

    def record_progress(progress):
        progress_calls.append(progress)

    select_robust_best(
        study=study,
        suggest_parameter_values=object(),
        incumbent_params={},
        objective_cfg=objective_config(
            environment_factory=FakeEnvironmentFactory(),
            hooks=ObjectiveHookFactory(
                "elise",
                cfg=InfraCfg(drive_study_dir=tmp_path / "drive", local_study_dir=tmp_path / "local"),
                window=2,
            ).with_training_progress(training_calls.append),
        ),
        extra_seeds=(1,),
        progress_fn=record_progress,
    )

    assert len(training_calls) == 1
    assert progress_calls[0] == RobustnessProgress(
        candidate_index=1, candidate_count=1, seed_index=1, seed_count=1, candidate_seed_scores=[[100.0]]
    )
    assert training_calls[0].episode_returns == [1.0]


def test_robustness_progress_accepts_checkpoint_summaries() -> None:
    progress = RobustnessProgress(
        candidate_index=1,
        candidate_count=1,
        seed_index=1,
        seed_count=1,
        candidate_seed_scores=[[200.0]],
        checkpoint_summaries=[{"mean": 250.0}],
    )

    assert progress.checkpoint_summaries == [{"mean": 250.0}]


def test_select_robust_best_rejects_empty_study() -> None:
    with pytest.raises(ValueError, match="no complete trials"):
        select_robust_best(
            study=FakeStudy(trials=[]),
            suggest_parameter_values=object(),
            incumbent_params={},
            objective_cfg=objective_config(environment_factory=FakeEnvironmentFactory()),
        )
