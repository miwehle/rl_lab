from pathlib import Path

import pytest

from hpo.lunar_lander import search_space


class FakeTrial:
    number = 7

    def __init__(self) -> None:
        self.suggestions = {}

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.suggestions[name] = value
        return value

    def suggest_float(self, name, low, high, *, log=False):
        value = low
        self.suggestions[name] = value
        return value

    def suggest_int(self, name, low, high, *, log=False):
        value = low
        self.suggestions[name] = value
        return value


def test_training_config_defines_training_search_space() -> None:
    trial = FakeTrial()

    config = search_space.training_config(trial, num_episodes=123)

    assert config.num_episodes == 123
    assert config.batch_size == 128
    assert config.eps_start == pytest.approx(0.9)
    assert config.eps_end == pytest.approx(0.01)
    assert config.eps_decay == 10_000
    assert config.learning_rate == pytest.approx(1e-5)
    assert config.gamma == pytest.approx(0.97)
    assert config.tau == pytest.approx(0.001)
    assert set(trial.suggestions) == {
        "batch_size",
        "eps_decay",
        "learning_rate",
        "gamma",
        "tau",
    }


def test_replay_memory_capacity_is_fixed_for_now() -> None:
    trial = FakeTrial()

    assert search_space.replay_memory_capacity(trial) == 10_000
    assert trial.suggestions == {}


def test_tuning_config_defines_tuning_search_space() -> None:
    trial = FakeTrial()
    output_dir = Path("hpo-output")

    config = search_space.tuning_config(trial, output_dir=output_dir)

    assert config.learning_starts == 1_000
    assert config.optimize_every == 1
    assert config.double_dqn is True
    assert config.save_best_checkpoint is False
    assert config.log_path == output_dir / "trial_0007" / "episodes.csv"
    assert set(trial.suggestions) == {
        "learning_starts",
        "optimize_every",
        "double_dqn",
    }
