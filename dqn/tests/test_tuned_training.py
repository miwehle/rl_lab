import csv
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from dqn.training import TrainingConfig
from dqn.tuned_training import TunedTrainer, TuningConfig


def training_config(**overrides) -> TrainingConfig:
    base = dict(
        num_episodes=1, batch_size=1, eps_start=1, eps_end=0,
        eps_decay=2500, learning_rate=1e-3,
    )
    return TrainingConfig(**(base | overrides))


def test_tuned_trainer_waits_for_warmup_and_optimizes_every_n_steps() -> None:
    env = gym.make("CartPole-v1")
    checkpoint_path = Path("dqn/tests/tuned_best.pt")
    checkpoint_path.unlink(missing_ok=True)

    try:
        trainer = TunedTrainer(
            env,
            seed=42,
            tuning_config=TuningConfig(
                learning_starts=8,
                optimize_every=4,
                checkpoint_window=2,
                checkpoint_path=checkpoint_path,
            ),
        )
        config = training_config(batch_size=4)

        for _ in range(config.batch_size):
            trainer.memory.push(None, None, None, None)

        trainer.steps_done = 7
        assert not trainer._should_optimize(config)

        trainer.steps_done = 8
        assert trainer._should_optimize(config)

        trainer.steps_done = 10
        assert not trainer._should_optimize(config)
    finally:
        checkpoint_path.unlink(missing_ok=True)
        env.close()


def test_tuned_training_logs_episode_metrics(tmp_path) -> None:
    env = gym.make("CartPole-v1")
    log_path = tmp_path / "training_log.csv"

    try:
        trainer = TunedTrainer(
            env,
            seed=42,
            tuning_config=TuningConfig(log_path=log_path),
        )
        config = training_config()

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
        "return",
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
    assert rows[0]["return"] == "10,0"
    assert rows[0]["mean_return"] == "10,0"
    assert rows[0]["best_mean_return"] == "10,0"

    timestamp = datetime.strptime(rows[1]["timestamp"], "%Y-%m-%d %H:%M:%S")
    assert timestamp.microsecond == 0
    assert "+" not in rows[1]["timestamp"]
    assert rows[1]["episode"] == "3"
    assert rows[1]["steps_done"] == "80"
    assert rows[1]["return"] == "40,0"
    assert rows[1]["mean_return"] == "23,3"
    assert rows[1]["best_mean_return"] == "23,3"
    assert rows[1]["epsilon"] == f"{trainer._exploration_rate(config):.3f}".replace(".", ",")


def test_tuned_trainer_saves_best_checkpoint() -> None:
    env = gym.make("CartPole-v1")
    checkpoint_path = Path("dqn/tests/tuned_best.pt")
    checkpoint_path.unlink(missing_ok=True)

    try:
        trainer = TunedTrainer(
            env,
            seed=42,
            tuning_config=TuningConfig(
                save_best_checkpoint=True,
                checkpoint_window=2,
                checkpoint_path=checkpoint_path,
            ),
        )
        config = training_config()

        trainer._after_episode([1.0], [10], config)
        assert checkpoint_path.exists()
        assert trainer.best_checkpoint_score == 1.0

        trainer._after_episode([1.0, 0.5], [10, 8], config)
        assert trainer.best_checkpoint_score == 1.0

        trainer._after_episode([1.0, 0.5, 2.0], [10, 8, 12], config)
        assert trainer.best_checkpoint_score == 1.25

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert checkpoint["version"] == 1
        assert set(checkpoint["trainer"]["q_net"]) == set(trainer.q_net.state_dict())
    finally:
        checkpoint_path.unlink(missing_ok=True)
        env.close()


def test_tuned_trainer_skips_checkpoint_below_min_score() -> None:
    env = gym.make("CartPole-v1")
    checkpoint_path = Path("dqn/tests/tuned_best.pt")
    checkpoint_path.unlink(missing_ok=True)

    try:
        trainer = TunedTrainer(
            env,
            seed=42,
            tuning_config=TuningConfig(
                save_best_checkpoint=True,
                checkpoint_window=2,
                checkpoint_min_score=10.0,
                checkpoint_path=checkpoint_path,
            ),
        )
        config = training_config()

        trainer._after_episode([9.0], [10], config)

        assert not checkpoint_path.exists()
        assert trainer.best_checkpoint_score == float("-inf")
    finally:
        checkpoint_path.unlink(missing_ok=True)
        env.close()


def test_tuned_trainer_does_not_save_checkpoint_by_default() -> None:
    env = gym.make("CartPole-v1")
    checkpoint_path = Path("dqn/tests/tuned_best.pt")
    checkpoint_path.unlink(missing_ok=True)

    try:
        trainer = TunedTrainer(
            env,
            seed=42,
            tuning_config=TuningConfig(
                checkpoint_window=2,
                checkpoint_path=checkpoint_path,
            ),
        )
        config = training_config()

        trainer._after_episode([1.0], [10], config)

        assert not checkpoint_path.exists()
        assert trainer.best_checkpoint_score == float("-inf")
        assert trainer.checkpoint_returns == [1.0]
    finally:
        checkpoint_path.unlink(missing_ok=True)
        env.close()
