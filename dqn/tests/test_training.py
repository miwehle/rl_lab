import gymnasium as gym
import pytest

from dqn.config import TrainingConfig
from dqn.training import TrainingResult, train
from helpers import model_hash, ema

# 5 s
def test_cartpole_training_smoke() -> None:
    env = gym.make("CartPole-v1")

    try:
        result = train(
            env,
            TrainingConfig(
                num_episodes=1,
                max_steps_per_episode=5,
                batch_size=2,
                seed=42,
            ),
        )
    finally:
        env.close()

    assert isinstance(result, TrainingResult)
    assert len(result.episode_returns) == 1
    assert result.episode_lengths == [5]
    assert result.device.type in {"cpu", "cuda", "mps"}


# 23 s
def test_cartpole_training() -> None:
    env = gym.make("CartPole-v1")

    config = TrainingConfig(
        num_episodes=100,
        seed=42
    )
    
    try:
        result = train(env, config)
    finally:
        env.close()

    emas = ema(result.episode_returns) 
    mh = model_hash(result.policy_net)
    n = config.num_episodes

    """
    print(result.episode_returns)
    print(result.episode_lengths)
    print(emas)
    print(mh)
    """

    # increasingly strict asserts
    assert isinstance(result, TrainingResult)
    assert len(result.episode_returns) == n
    assert result.device.type in {"cpu", "cuda", "mps"}
    assert emas[n-1] > 150
    assert emas[n-1] == pytest.approx(167.4, abs=0.1)
    assert mh == "631a91e7115d4ae23f17ecc44f0ff29afa6b7b9491507756a6b09728677ee6ac"
