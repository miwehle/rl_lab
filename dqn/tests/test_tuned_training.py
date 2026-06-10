from pathlib import Path

import gymnasium as gym
import torch

from dqn.tuned_training import TunedTrainer, TunedTrainingConfig


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
        config = TunedTrainingConfig(checkpoint_path=checkpoint_path)

        trainer._after_episode([1.0], [10], config)
        assert checkpoint_path.exists()
        assert trainer.best_checkpoint_return == 1.0

        trainer._after_episode([1.0, 0.5], [10, 8], config)
        assert trainer.best_checkpoint_return == 1.0

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        assert set(checkpoint) == set(trainer.q_net.state_dict())
    finally:
        checkpoint_path.unlink(missing_ok=True)
        env.close()
