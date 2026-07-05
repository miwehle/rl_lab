from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from dqn.model import DQN
from hpo.evaluation.checkpoint_robustness import (
    checkpoint_scores,
    evaluate_checkpoint_robustness,
    robustness_over_all_worlds,
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

    assert len(results) == 1
    result = results[0]
    assert result["trial_number"] == 1
    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["source_score"] == pytest.approx(200.0)
    assert result["robust_score"] == pytest.approx(15.0)
    assert result["score"] == pytest.approx(15.0)
    assert result["world_scores"] == {"earth": pytest.approx(10.0), "venus": pytest.approx(20.0)}
    assert result["eval_episodes"] == 3
    assert result["checkpoint_summary"] == {
        "candidate": 1,
        "trial_number": 1,
        "source_score": pytest.approx(200.0),
        "checkpoint_path": str(checkpoint_path),
        "episodes_per_world": 3,
        "episodes": 6,
        "mean": pytest.approx(15.0),
        "median": pytest.approx(15.0),
        "min": pytest.approx(10.0),
        "q05": pytest.approx(10.0),
        "q25": pytest.approx(10.0),
        "q75": pytest.approx(20.0),
        "q95": pytest.approx(20.0),
        "max": pytest.approx(20.0),
        "world_scores": {"earth": pytest.approx(10.0), "venus": pytest.approx(20.0)},
    }
    assert study.user_attrs["checkpoint_robustness"] == results
    assert progress_calls[-1].title == "Checkpoint Robustness Evaluation"
    assert progress_calls[-1].candidate_seed_scores == [[200.0, 15.0]]
    assert progress_calls[-1].checkpoint_summaries == [result["checkpoint_summary"]]


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
        "hpo.evaluation.checkpoint_robustness._tqdm",
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


def test_robustness_over_all_worlds_returns_candidate_summary(tmp_path) -> None:
    checkpoint_path = tmp_path / "trial_0001_eval_best.pt"
    save_checkpoint(
        DQN(1, 2),
        checkpoint_path,
        {"score": 200.0, "episode": 1000, "window": None},
    )

    summary = robustness_over_all_worlds(
        checkpoint_path,
        objective_config(
            environment_factory=FakeEnvironmentFactory(),
            device="cpu",
        ),
        episodes=2,
        progress=False,
    )

    assert summary == {
        "checkpoint_path": str(checkpoint_path),
        "episodes_per_world": 2,
        "episodes": 4,
        "mean": pytest.approx(15.0),
        "median": pytest.approx(15.0),
        "min": pytest.approx(10.0),
        "q05": pytest.approx(10.0),
        "q25": pytest.approx(10.0),
        "q75": pytest.approx(20.0),
        "q95": pytest.approx(20.0),
        "max": pytest.approx(20.0),
        "world_scores": {"earth": pytest.approx(10.0), "venus": pytest.approx(20.0)},
    }


def test_score_summary_returns_notebook_quantiles() -> None:
    scores = pd.DataFrame({
        "world": ["venus", "venus", "venus", "earth", "earth", "earth"],
        "score": [100.0, 110.0, 120.0, 0.0, 10.0, 20.0],
    })

    summary = score_summary(scores)

    assert list(summary.index) == ["venus", "earth"]
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
