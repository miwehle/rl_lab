import time

import gymnasium as gym
import pytest

from dqn.model import DQN
from dqn.training import Trainer, TrainingConfig, TrainingResult
from dqn_helpers import model_hash, ema

MAX_SMOKE_TEST_SECONDS = 6.0
MAX_TEST_SECONDS = 30.0


def training_config(**overrides) -> TrainingConfig:
    base = dict(
        num_episodes=50, batch_size=128, eps_start=0.9, eps_end=0.01,
        eps_decay=2500, learning_rate=3e-4,
    )
    return TrainingConfig(**(base | overrides))


# 5 s
@pytest.mark.timeout(MAX_SMOKE_TEST_SECONDS)
def test_cartpole_training_smoke() -> None:
    env = gym.make("CartPole-v1")

    try:
        trainer = Trainer(env, seed=42)
        result = trainer.train(
            training_config(num_episodes=1, batch_size=2),
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
        first_result = trainer.train(training_config(num_episodes=1, batch_size=2))
        steps_after_first_run = trainer.steps_done

        second_result = trainer.train(
            training_config(
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


# 23 s
@pytest.mark.timeout(MAX_TEST_SECONDS)
def test_cartpole_training() -> None:
    env = gym.make("CartPole-v1")

    config = training_config(num_episodes=100)
    
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
    assert emas[episodes-1] == pytest.approx(171.5, abs=0.1)
    assert model_hash(result.q_net) == "3102a7f2deee801ec5718c2d7e4f420cd2919d5bb315eac0d5170752ee8d3e4f"
