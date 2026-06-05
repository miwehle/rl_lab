import gymnasium as gym
import pytest

from dqn.config import TrainingConfig
from dqn.training import Trainer, TrainingResult
from helpers import model_hash, ema

# 5 s
def test_cartpole_training_smoke() -> None:
    env = gym.make("CartPole-v1")

    try:
        trainer = Trainer(env, seed=42)
        result = trainer.train(
            TrainingConfig(
                num_episodes=1,
                max_steps_per_episode=5,
                batch_size=2,
            ),
        )
    finally:
        env.close()

    assert isinstance(result, TrainingResult)
    assert len(result.episode_returns) == 1
    assert result.episode_lengths == [5]
    assert result.device.type in {"cpu", "cuda", "mps"}


def test_training_can_continue_with_another_config() -> None:
    env = gym.make("CartPole-v1")

    try:
        trainer = Trainer(env, seed=42)
        first_result = trainer.train(
            TrainingConfig(
                num_episodes=1,
                max_steps_per_episode=2,
                batch_size=2,
            )
        )
        steps_after_first_run = trainer.steps_done

        second_result = trainer.train(
            TrainingConfig(
                num_episodes=2,
                max_steps_per_episode=3,
                batch_size=2,
                learning_rate=1e-4,
            )
        )
    finally:
        env.close()

    assert first_result.episode_lengths == [2]
    assert second_result.episode_lengths == [3, 3]
    assert trainer.steps_done == steps_after_first_run + 6
    assert trainer.memory is not None
    assert len(trainer.memory) == 8
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-4


# 23 s
def test_cartpole_training() -> None:
    env = gym.make("CartPole-v1")

    config = TrainingConfig(
        num_episodes=100,
    )
    
    try:
        trainer = Trainer(env, seed=42)
        result = trainer.train(config)
    finally:
        env.close()

    emas = ema(result.episode_returns) 
    episodes = config.num_episodes

    """
    print(result.episode_returns)
    print(result.episode_lengths)
    print(emas)
    print(mh)
    """

    # increasingly strict asserts
    assert isinstance(result, TrainingResult)
    assert len(result.episode_returns) == episodes
    assert result.device.type in {"cpu", "cuda", "mps"}
    assert emas[episodes-1] > 150
    assert emas[episodes-1] == pytest.approx(167.4, abs=0.1)
    assert model_hash(result.policy_net) == "631a91e7115d4ae23f17ecc44f0ff29afa6b7b9491507756a6b09728677ee6ac"
