from pathlib import Path

import gymnasium as gym
import pytest
import torch

from dqn.tuned_training import TunedTrainer, TunedTrainingConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_starts": -1},
        {"optimize_every": 0},
        {"checkpoint_window": 0},
        {"batch_size": 0},
    ],
)
def test_tuned_training_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        TunedTrainingConfig(**kwargs)


def test_tuned_training_config_warns_when_epsilon_decays_before_learning_starts() -> None:
    with pytest.warns(UserWarning, match="eps_decay may be too small"):
        TunedTrainingConfig(learning_starts=1000, eps_decay=500)


def test_tuned_trainer_waits_for_warmup_and_optimizes_every_n_steps() -> None:
    env = gym.make("CartPole-v1")
    checkpoint_path = Path("dqn/tests/tuned_best.pt")
    checkpoint_path.unlink(missing_ok=True)

    try:
        trainer = TunedTrainer(env, seed=42)
        config = TunedTrainingConfig(
            batch_size=4,
            learning_starts=8,
            optimize_every=4,
            checkpoint_window=2,
            checkpoint_path=checkpoint_path,
        )

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


def test_tuned_trainer_saves_best_checkpoint() -> None:
    env = gym.make("CartPole-v1")
    checkpoint_path = Path("dqn/tests/tuned_best.pt")
    checkpoint_path.unlink(missing_ok=True)

    try:
        trainer = TunedTrainer(env, seed=42)
        config = TunedTrainingConfig(
            save_best_checkpoint=True,
            checkpoint_window=2,
            checkpoint_path=checkpoint_path,
        )

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


def test_tuned_trainer_does_not_save_checkpoint_by_default() -> None:
    env = gym.make("CartPole-v1")
    checkpoint_path = Path("dqn/tests/tuned_best.pt")
    checkpoint_path.unlink(missing_ok=True)

    try:
        trainer = TunedTrainer(env, seed=42)
        config = TunedTrainingConfig(checkpoint_window=2, checkpoint_path=checkpoint_path)

        trainer._after_episode([1.0], [10], config)

        assert not checkpoint_path.exists()
        assert trainer.best_checkpoint_score == float("-inf")
        assert trainer.checkpoint_returns == [1.0]
    finally:
        checkpoint_path.unlink(missing_ok=True)
        env.close()
