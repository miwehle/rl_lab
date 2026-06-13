import pytest

from hpo.evaluation import pruning
from hpo.evaluation.pruning import PruningConfig, create_pruning_callback


class FakeTrial:
    def __init__(self) -> None:
        self.reports = []
        self.user_attrs = {}

    def report(self, value, step) -> None:
        self.reports.append((value, step))

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


def test_create_pruning_callback_returns_none_when_disabled() -> None:
    assert (
        create_pruning_callback(FakeTrial(), None, num_episodes=3, score_window=2)
        is None
    )


def test_pruning_callback_reports_score_after_start_episode() -> None:
    trial = FakeTrial()
    callback = create_pruning_callback(
        trial,
        PruningConfig(
            start_episode=3,
            min_score=10.0,
        ),
        num_episodes=5,
        score_window=2,
    )

    callback((1.0, 20.0))
    assert trial.reports == []

    callback((1.0, 20.0, 30.0))
    assert trial.reports == [(25.0, 3)]


def test_pruning_callback_prunes_below_min_score(monkeypatch) -> None:
    monkeypatch.setattr(pruning, "_trial_pruned", RuntimeError)
    callback = create_pruning_callback(
        trial := FakeTrial(),
        PruningConfig(
            start_episode=3,
            min_score=100.0,
        ),
        num_episodes=5,
        score_window=2,
    )

    with pytest.raises(RuntimeError, match="below 100.0"):
        callback((1.0, 20.0, 30.0))
    assert trial.user_attrs == {
        "pruned_at_episode": 3,
        "episodes_saved_by_pruning": 2,
    }
