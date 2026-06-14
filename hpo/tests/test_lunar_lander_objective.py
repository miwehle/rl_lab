from dataclasses import dataclass
from pathlib import Path

import pytest

from dqn.training import TrainingConfig
from dqn.training import TrainingResult
from dqn.tuned_training import TuningConfig
from hpo.evaluation.pruning import PruningConfig
from hpo.lunar_lander.objective import create_objective


class FakeTrial:
    number = 3

    def __init__(self) -> None:
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, *, log=False):
        return low

    def suggest_int(self, name, low, high, *, log=False):
        return low

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


class FakeEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeSearchSpace:
    def __init__(self) -> None:
        self.calls = []

    def training_config(self, trial, num_episodes: int) -> TrainingConfig:
        self.calls.append(("training_config", trial, num_episodes))
        return TrainingConfig(
            num_episodes=num_episodes,
            batch_size=64,
            eps_start=0.7,
            eps_end=0.02,
            eps_decay=1234,
            learning_rate=5e-4,
        )

    def replay_memory_capacity(self, trial) -> int:
        self.calls.append(("replay_memory_capacity", trial))
        return 12_345

    def tuning_config(self, trial, *, output_dir: Path | None) -> TuningConfig:
        self.calls.append(("tuning_config", trial, output_dir))
        log_path = None
        if output_dir is not None:
            log_path = output_dir / f"fake_trial_{trial.number}.csv"

        return TuningConfig(
            learning_starts=77,
            optimize_every=3,
            double_dqn=True,
            save_best_checkpoint=False,
            checkpoint_min_score=0.0,
            checkpoint_min_score_delta=0.0,
            log_path=log_path,
        )


@dataclass
class TrainerCall:
    env: FakeEnv
    seed: int | None
    device: object
    replay_memory_capacity: int
    tuning_config: object
    after_episode_callback: object
    training_config: object | None = None


def test_lunar_lander_objective_trains_trial_and_returns_score() -> None:
    calls = []
    envs = []
    output_dir = Path("hpo-output")
    search_space = FakeSearchSpace()

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
            after_episode_callback,
        ) -> None:
            calls.append(
                TrainerCall(
                    env,
                    seed,
                    device,
                    replay_memory_capacity,
                    tuning_config,
                    after_episode_callback,
                )
            )

        def train(self, training_config):
            calls[-1].training_config = training_config
            return TrainingResult(
                q_net=None,
                episode_returns=[10.0, 20.0, 30.0],
                episode_lengths=[1, 1, 1],
            )

    objective = create_objective(
        search_space=search_space,
        num_episodes=12,
        score_window=2,
        seed=100,
        output_dir=output_dir,
        env_factory=env_factory,
        trainer_factory=FakeTrainer,
    )

    trial = FakeTrial()
    score = objective(trial)

    assert search_space.calls == [
        ("training_config", trial, 12),
        ("replay_memory_capacity", trial),
        ("tuning_config", trial, output_dir),
    ]
    assert score == pytest.approx(25.0)
    assert trial.user_attrs == {
        "best_window_mean": pytest.approx(25.0),
        "best_window_start_episode": 2,
        "best_window_end_episode": 3,
    }
    assert envs[0].closed
    assert calls[0].seed == 103
    assert calls[0].replay_memory_capacity == 12_345
    assert calls[0].training_config.num_episodes == 12
    assert calls[0].training_config.learning_rate == pytest.approx(5e-4)
    assert calls[0].training_config.batch_size == 64
    assert calls[0].tuning_config.learning_starts == 77
    assert calls[0].tuning_config.optimize_every == 3
    assert calls[0].tuning_config.double_dqn is True
    assert calls[0].tuning_config.save_best_checkpoint is False
    assert calls[0].tuning_config.log_path == output_dir / "fake_trial_3.csv"
    assert calls[0].after_episode_callback is None


def test_lunar_lander_objective_passes_pruning_callback_to_trainer() -> None:
    calls = []

    class FakeTrainer:
        def __init__(
            self,
            _env,
            *,
            seed,
            device,
            replay_memory_capacity,
            tuning_config,
            after_episode_callback,
        ) -> None:
            calls.append(after_episode_callback)

        def train(self, _training_config):
            return TrainingResult(
                q_net=None,
                episode_returns=[10.0, 20.0, 30.0],
                episode_lengths=[1, 1, 1],
            )

    objective = create_objective(
        search_space=FakeSearchSpace(),
        num_episodes=12,
        score_window=2,
        pruning_config=PruningConfig(),
        env_factory=lambda _env_id: FakeEnv(),
        trainer_factory=FakeTrainer,
    )

    objective(FakeTrial())

    assert calls[0] is not None
