import csv
from datetime import datetime
import time

import gymnasium as gym
import pytest

from dqn.model import DQN
from dqn.training import Trainer, TrainingConfig, TrainingResult
from helpers import model_hash, ema

MAX_SMOKE_TEST_SECONDS = 6.0
MAX_TEST_SECONDS = 30.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"eps_decay": 0},
        {"num_episodes": 0},
        {"gamma": -0.1},
        {"eps_start": 1.1},
        {"eps_end": -0.1},
        {"tau": 1.1},
        {"learning_rate": 0},
    ],
)
def test_training_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


# 5 s
@pytest.mark.timeout(MAX_SMOKE_TEST_SECONDS)
def test_cartpole_training_smoke() -> None:
    env = gym.make("CartPole-v1")

    try:
        trainer = Trainer(env, seed=42)
        result = trainer.train(
            TrainingConfig(
                num_episodes=1,
                batch_size=2,
            ),
        )
    finally:
        env.close()

    assert isinstance(result, TrainingResult)
    assert len(result.episode_returns) == 1
    assert result.episode_lengths[0] > 0
    assert trainer.device.type in {"cpu", "cuda", "mps"}


def test_training_can_continue_with_another_config() -> None:
    env = gym.make("CartPole-v1")

    try:
        trainer = Trainer(env, seed=42)
        first_result = trainer.train(
            TrainingConfig(
                num_episodes=1,
                batch_size=2,
            )
        )
        steps_after_first_run = trainer.steps_done

        second_result = trainer.train(
            TrainingConfig(
                num_episodes=2,
                batch_size=2,
                learning_rate=1e-4,
            )
        )
    finally:
        env.close()

    assert len(first_result.episode_lengths) == 1
    assert len(second_result.episode_lengths) == 2
    assert all(length > 0 for length in first_result.episode_lengths)
    assert all(length > 0 for length in second_result.episode_lengths)
    assert trainer.steps_done == steps_after_first_run + sum(second_result.episode_lengths)
    assert trainer.memory is not None
    assert len(trainer.memory) == trainer.steps_done
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_training_logs_episode_metrics(tmp_path) -> None:
    env = gym.make("CartPole-v1")
    log_path = tmp_path / "training_log.csv"

    try:
        trainer = Trainer(env, seed=42)
        config = TrainingConfig(log_path=log_path)

        trainer.steps_done = 25
        trainer._after_episode([10.0], [10], config)

        trainer.steps_done = 80
        trainer._after_episode([10.0, 20.0, 40.0], [10, 12, 14], config)
    finally:
        env.close()

    with log_path.open(encoding="utf-8", newline="") as log_file:
        reader = csv.DictReader(log_file, delimiter=";")
        rows = list(reader)

    assert reader.fieldnames == [
        "timestamp",
        "episode",
        "steps_done",
        "mean_return",
        "best_mean_return",
        "epsilon",
    ]
    assert len(rows) == 2

    timestamp = datetime.strptime(rows[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
    assert timestamp.microsecond == 0
    assert "+" not in rows[0]["timestamp"]
    assert rows[0]["episode"] == "1"
    assert rows[0]["steps_done"] == "25"
    assert rows[0]["mean_return"] == "10,0"
    assert rows[0]["best_mean_return"] == "10,0"

    timestamp = datetime.strptime(rows[1]["timestamp"], "%Y-%m-%d %H:%M:%S")
    assert timestamp.microsecond == 0
    assert "+" not in rows[1]["timestamp"]
    assert rows[1]["episode"] == "3"
    assert rows[1]["steps_done"] == "80"
    assert rows[1]["mean_return"] == "23,3"
    assert rows[1]["best_mean_return"] == "23,3"
    assert rows[1]["epsilon"] == f"{trainer._exploration_rate(config):.3f}".replace(".", ",")


# 23 s
@pytest.mark.timeout(MAX_TEST_SECONDS)
def test_cartpole_training() -> None:
    env = gym.make("CartPole-v1")

    config = TrainingConfig(
        num_episodes=100,
    )
    
    try:
        trainer = Trainer(env, model_factory=DQN, seed=42)
        start_time = time.perf_counter()
        result = trainer.train(config)
        elapsed_seconds = time.perf_counter() - start_time
    finally:
        env.close()

    emas = ema(result.episode_returns) 
    episodes = config.num_episodes

    """
    print(result.episode_returns)
    print(result.episode_lengths)
    print(emas)
    print(f"elapsed_seconds = {elapsed_seconds:.2f}")
    """
    
    # increasingly strict asserts
    assert isinstance(result, TrainingResult)
    assert len(result.episode_returns) == episodes
    assert trainer.device.type in {"cpu", "cuda", "mps"}
    assert elapsed_seconds <= MAX_TEST_SECONDS - 5.0
    assert emas[episodes-1] > 150
    assert emas[episodes-1] == pytest.approx(171.0, abs=0.1)
    assert model_hash(result.q_net) == "0d99a213bfe0afc6c5e902f000e39e79cd0f67c6313090a014a2056d58b16707"
