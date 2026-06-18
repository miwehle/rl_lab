import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
import torch

from dqn.training import TrainingResult
from dqn.vector_training import (
    VectorReplayMemory,
    VectorTrainer,
    VectorTrainingConfig,
    VectorTrainingResult,
)


def vector_env(num_envs: int):
    return SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(num_envs)])


def vector_training_config(**overrides) -> VectorTrainingConfig:
    base = dict(
        num_episodes=4,
        batch_size=4,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=250,
        learning_rate=1e-3,
        learning_starts=0,
        optimize_every=4,
    )
    return VectorTrainingConfig(**(base | overrides))


def test_vector_replay_memory_samples_tensors_on_device() -> None:
    memory = VectorReplayMemory(capacity=8, observation_shape=(4,), seed=42)
    states = np.zeros((3, 4), dtype=np.float32)
    next_states = np.ones((3, 4), dtype=np.float32)

    memory.push_batch(
        states,
        np.array([0, 1, 0]),
        next_states,
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([False, True, False]),
    )

    batch = memory.sample(batch_size=2, device=torch.device("cpu"))

    assert len(memory) == 3
    assert batch.states.shape == (2, 4)
    assert batch.actions.shape == (2, 1)
    assert batch.next_states.shape == (2, 4)
    assert batch.rewards.shape == (2,)
    assert batch.terminated.shape == (2,)


def test_vector_training_smoke() -> None:
    env = vector_env(num_envs=2)

    try:
        trainer = VectorTrainer(env, seed=42)
        result = trainer.train(vector_training_config())
    finally:
        env.close()

    assert isinstance(result, TrainingResult)
    assert isinstance(result, VectorTrainingResult)
    assert len(result.episode_returns) == 4
    assert len(result.episode_lengths) == 4
    assert len(result.episode_epsilons) == 4
    assert result.env_steps > 0
    assert result.optimizer_updates > 0
    assert all(length > 0 for length in result.episode_lengths)
    assert all(0.05 <= epsilon <= 1.0 for epsilon in result.episode_epsilons)
    assert trainer.device.type in {"cpu", "cuda", "mps"}
    assert len(trainer.memory) > 0


def test_vector_training_accepts_plotter() -> None:
    env = vector_env(num_envs=2)
    plot_calls = []

    class Plotter:
        def plot_returns(self, returns, show_result=False, epsilons=None) -> None:
            plot_calls.append((list(returns), show_result, epsilons))

    try:
        trainer = VectorTrainer(env, seed=42)
        trainer.train(vector_training_config(num_episodes=2), plotter=Plotter())
    finally:
        env.close()

    assert plot_calls
    assert len(plot_calls[-1][0]) == 2
    assert plot_calls[-1][2] is not None


def test_vector_trainer_optimizes_for_each_crossed_interval() -> None:
    env = vector_env(num_envs=4)
    calls = []

    try:
        trainer = VectorTrainer(env, seed=42)
        observations = np.zeros((8, 4), dtype=np.float32)
        trainer.memory.push_batch(
            observations,
            np.zeros(8, dtype=np.int64),
            observations,
            np.zeros(8, dtype=np.float32),
            np.zeros(8, dtype=np.bool_),
        )
        trainer.steps_done = 8

        trainer._optimize_model = calls.append
        trainer._soft_target_update = lambda _tau: None
        trainer._optimize_due(
            vector_training_config(batch_size=1, optimize_every=2),
            previous_steps=0,
        )
    finally:
        env.close()

    assert len(calls) == 4
    assert trainer.optimizer_updates == 0
