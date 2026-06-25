from dataclasses import dataclass, field

import pytest

from hpo import robust_selection as robust_selection_module
from hpo.robust_selection import select_robust_best
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

    monkeypatch.setattr(
        robust_selection_module,
        "create_objective",
        fake_create_objective,
    )
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
