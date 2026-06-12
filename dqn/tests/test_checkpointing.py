from pathlib import Path
import random

import gymnasium as gym
import pytest
import torch

from dqn.checkpointing import load_checkpoint, save_checkpoint
from dqn.training import TrainingConfig
from dqn.tuned_training import TunedTrainer, TuningConfig
from helpers import model_hash


def test_checkpoint_restores_trainer_state() -> None:
    checkpoint_path = Path("dqn/tests/checkpoint.pt")
    checkpoint_path.unlink(missing_ok=True)
    env = gym.make("CartPole-v1")
    trainer = TunedTrainer(
        env,
        seed=42,
        tuning_config=TuningConfig(
            learning_starts=0,
            optimize_every=1,
        ),
    )

    try:
        trainer.train(
            TrainingConfig(
                num_episodes=1,
                batch_size=2,
                eps_start=0.9,
                eps_end=0.01,
                eps_decay=2500,
                learning_rate=1e-4,
            ),
        )
        trainer.best_checkpoint_score = 123.0
        trainer.checkpoint_returns = [1.0, 2.0, 3.0]

        q_net_hash = model_hash(trainer.q_net)
        target_net_hash = model_hash(trainer.target_net)
        steps_done = trainer.steps_done
        replay_memory_length = len(trainer.memory)
        replay_memory_capacity = trainer.memory.memory.maxlen

        save_checkpoint(trainer, checkpoint_path)

        expected_python_random = random.random()
        expected_torch_random = torch.rand(3)
        expected_action_sample = env.action_space.sample()
        expected_observation_sample = env.observation_space.sample()
        expected_env_random = env.np_random.random()
    finally:
        env.close()

    restored_env = gym.make("CartPole-v1")
    restored_trainer = TunedTrainer(restored_env, seed=100)

    try:
        load_checkpoint(restored_trainer, checkpoint_path)

        assert model_hash(restored_trainer.q_net) == q_net_hash
        assert model_hash(restored_trainer.target_net) == target_net_hash
        assert restored_trainer.steps_done == steps_done
        assert len(restored_trainer.memory) == replay_memory_length
        assert restored_trainer.memory.memory.maxlen == replay_memory_capacity
        assert restored_trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert restored_trainer.optimizer.state_dict()["state"]
        assert restored_trainer.best_checkpoint_score == 123.0
        assert restored_trainer.checkpoint_returns == [1.0, 2.0, 3.0]

        assert random.random() == expected_python_random
        torch.testing.assert_close(torch.rand(3), expected_torch_random)
        assert restored_env.action_space.sample() == expected_action_sample
        assert restored_env.observation_space.sample() == pytest.approx(
            expected_observation_sample,
        )
        assert restored_env.np_random.random() == expected_env_random
    finally:
        checkpoint_path.unlink(missing_ok=True)
        restored_env.close()


def test_checkpoint_loads_list_encoded_torch_rng_state() -> None:
    checkpoint_path = Path("dqn/tests/list_rng_checkpoint.pt")
    checkpoint_path.unlink(missing_ok=True)
    env = gym.make("CartPole-v1")
    trainer = TunedTrainer(env, seed=42)

    try:
        save_checkpoint(trainer, checkpoint_path)
        expected_torch_random = torch.rand(3)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["rng"]["torch"] = checkpoint["rng"]["torch"].tolist()
        torch.save(checkpoint, checkpoint_path)
    finally:
        env.close()

    restored_env = gym.make("CartPole-v1")
    restored_trainer = TunedTrainer(restored_env, seed=100)

    try:
        load_checkpoint(restored_trainer, checkpoint_path)

        torch.testing.assert_close(torch.rand(3), expected_torch_random)
    finally:
        checkpoint_path.unlink(missing_ok=True)
        restored_env.close()
