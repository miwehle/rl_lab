"""Train a DQN agent on CartPole-v1."""

import gymnasium as gym

from dqn.training import Trainer, TrainingConfig


def main() -> None:
    env = gym.make("CartPole-v1")
    config = TrainingConfig()
    trainer = Trainer(env)
    result = trainer.train(config)
    env.close()

    final_return = result.episode_returns[-1] if result.episode_returns else 0.0
    print(f"Complete. Episodes: {len(result.episode_returns)}. Final return: {final_return:.1f}")


if __name__ == "__main__":
    main()
