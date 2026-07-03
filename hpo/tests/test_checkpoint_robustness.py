from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from dqn.model import DQN
from hpo.checkpoint_robustness import (
    checkpoint_scores,
    evaluate_checkpoint_robustness,
    score_summary,
)
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


def test_checkpoint_scores_returns_episode_scores_by_world(tmp_path) -> None:
    checkpoint_path = tmp_path / "trial_0001_eval_best.pt"
    save_checkpoint(
        DQN(1, 2),
        checkpoint_path,
        {"score": 200.0, "episode": 1000, "window": None},
    )

    scores = checkpoint_scores(
        checkpoint_path,
        objective_config(
            environment_factory=FakeEnvironmentFactory(),
            device="cpu",
        ),
        episodes=2,
        progress=False,
    )

    assert scores.to_dict("records") == [
        {"world": "earth", "episode": 0, "score": 10.0},
        {"world": "earth", "episode": 1, "score": 10.0},
        {"world": "venus", "episode": 0, "score": 20.0},
        {"world": "venus", "episode": 1, "score": 20.0},
    ]


def test_checkpoint_scores_uses_progress_bar_when_enabled(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "trial_0001_eval_best.pt"
    save_checkpoint(
        DQN(1, 2),
        checkpoint_path,
        {"score": 200.0, "episode": 1000, "window": None},
    )
    progress_calls = []

    def fake_tqdm(items, *, total, desc):
        progress_calls.append({"total": total, "desc": desc})
        return items

    monkeypatch.setattr(
        "hpo.checkpoint_robustness._tqdm",
        lambda: fake_tqdm,
    )

    checkpoint_scores(
        checkpoint_path,
        objective_config(
            environment_factory=FakeEnvironmentFactory(),
            device="cpu",
        ),
        episodes=2,
    )

    assert progress_calls == [{"total": 4, "desc": "Evaluating checkpoint"}]


def test_score_summary_returns_notebook_quantiles() -> None:
    scores = pd.DataFrame({
        "world": ["earth", "earth", "earth", "venus", "venus", "venus"],
        "score": [0.0, 10.0, 20.0, 100.0, 110.0, 120.0],
    })

    summary = score_summary(scores)

    assert list(summary.columns) == [
        "episodes",
        "mean",
        "std",
        "min",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "max",
    ]
    assert summary.loc["earth", "episodes"] == 3
    assert summary.loc["earth", "mean"] == pytest.approx(10.0)
    assert summary.loc["earth", "q05"] == pytest.approx(1.0)
    assert summary.loc["venus", "median"] == pytest.approx(110.0)
