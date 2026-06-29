from dataclasses import dataclass, field

import numpy as np
import pytest

from dqn.model import DQN
from hpo.checkpoint_robustness import evaluate_checkpoint_robustness
from hpo.checkpointing import save_checkpoint
from common import objective_config


@dataclass
class FakeState:
    name: str = "COMPLETE"


@dataclass
class FakeTrial:
    number: int
    value: float
    user_attrs: dict
    state: FakeState = field(default_factory=FakeState)


@dataclass
class FakeStudy:
    trials: list[FakeTrial]
    user_attrs: dict = field(default_factory=dict)

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


class FakeActionSpace:
    n = 2


class FakeEnv:
    action_space = FakeActionSpace()

    def __init__(self, reward: float) -> None:
        self.reward = reward

    def reset(self, *, seed=None):
        return np.array([0.0], dtype=np.float32), {}

    def step(self, _action):
        return np.array([0.0], dtype=np.float32), self.reward, True, False, {}

    def close(self) -> None:
        pass


class FakeEnvironmentFactory:
    def evaluation_envs(self):
        return {
            "earth": lambda: FakeEnv(10.0),
            "venus": lambda: FakeEnv(20.0),
        }


def test_evaluate_checkpoint_robustness_scores_top_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "trial_0001_eval_best.pt"
    save_checkpoint(
        DQN(1, 2),
        checkpoint_path,
        {"score": 200.0, "episode": 1000, "window": None},
    )
    study = FakeStudy(
        trials=[
            FakeTrial(
                0,
                300.0,
                {"evaluation_checkpoint_score": 50.0},
            ),
            FakeTrial(
                1,
                100.0,
                {
                    "evaluation_checkpoint_path": str(checkpoint_path),
                    "evaluation_checkpoint_score": 200.0,
                },
            ),
        ],
    )
    progress_calls = []

    results = evaluate_checkpoint_robustness(
        study=study,
        objective_cfg=objective_config(
            environment_factory=FakeEnvironmentFactory(),
            device="cpu",
        ),
        top_n=1,
        eval_episodes=3,
        progress_fn=progress_calls.append,
    )

    assert results == [
        {
            "trial_number": 1,
            "checkpoint_path": str(checkpoint_path),
            "source_score": 200.0,
            "robust_score": 15.0,
            "score": 107.5,
            "world_scores": {"earth": 10.0, "venus": 20.0},
            "eval_episodes": 3,
        }
    ]
    assert study.user_attrs["checkpoint_robustness"] == results
    assert progress_calls[-1].title == "Checkpoint Robustness Evaluation"
    assert progress_calls[-1].candidate_seed_scores == [[200.0, 15.0]]
