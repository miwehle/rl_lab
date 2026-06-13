from dataclasses import dataclass
from pathlib import Path

import pytest

from dqn.training import TrainingResult
from hpo.lunar_lander_objective import create_lunar_lander_objective, mean_last


class FakeTrial:
    number = 3

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, *, log=False):
        return low

    def suggest_int(self, name, low, high, *, log=False):
        return low


class FakeEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@dataclass
class TrainerCall:
    env: FakeEnv
    seed: int | None
    device: object
    replay_memory_capacity: int
    tuning_config: object
    training_config: object | None = None


def test_mean_last_uses_tail_window() -> None:
    assert mean_last([1.0, 2.0, 5.0], 2) == pytest.approx(3.5)


def test_mean_last_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="values must not be empty"):
        mean_last([], 5)


def test_lunar_lander_objective_trains_trial_and_returns_score() -> None:
    calls = []
    envs = []
    output_dir = Path("hpo-output")

    def env_factory(env_id):
        assert env_id == "LunarLander-v3"
        env = FakeEnv()
        envs.append(env)
        return env

    class FakeTrainer:
        def __init__(
            self,
            env,
            *,
            seed,
            device,
            replay_memory_capacity,
            tuning_config,
        ) -> None:
            calls.append(
                TrainerCall(
                    env,
                    seed,
                    device,
                    replay_memory_capacity,
                    tuning_config,
                )
            )

        def train(self, training_config):
            calls[-1].training_config = training_config
            return TrainingResult(
                q_net=None,
                episode_returns=[10.0, 20.0, 30.0],
                episode_lengths=[1, 1, 1],
            )

    objective = create_lunar_lander_objective(
        num_episodes=12,
        score_window=2,
        seed=100,
        output_dir=output_dir,
        env_factory=env_factory,
        trainer_factory=FakeTrainer,
    )

    trial = FakeTrial()
    score = objective(trial)

    assert score == pytest.approx(25.0)
    assert envs[0].closed
    assert calls[0].seed == 103
    assert calls[0].replay_memory_capacity == 10_000
    assert calls[0].training_config.num_episodes == 12
    assert calls[0].training_config.learning_rate == pytest.approx(1e-5)
    assert calls[0].training_config.batch_size == 128
    assert calls[0].tuning_config.learning_starts == 1_000
    assert calls[0].tuning_config.optimize_every == 1
    assert calls[0].tuning_config.double_dqn is True
    assert calls[0].tuning_config.save_best_checkpoint is False
    assert calls[0].tuning_config.log_path == output_dir / "trial_0003" / "episodes.csv"
