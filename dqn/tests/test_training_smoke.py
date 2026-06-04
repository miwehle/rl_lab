import gymnasium as gym

from dqn.config import TrainingConfig
from dqn.training import TrainingResult, train


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

