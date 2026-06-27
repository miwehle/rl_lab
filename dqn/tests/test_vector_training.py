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
    _early_stopping_score,
    _should_extend_training,
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


def test_vector_training_updates_plotter_target_when_training_extends(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dqn.vector_training._should_extend_training",
        lambda episode_returns, **_kwargs: len(episode_returns) == 2,
    )
    env = vector_env(num_envs=2)
    plot_calls = []

    class Plotter:
        target_episodes = 2

        def plot_returns(self, returns, **_kwargs) -> None:
            plot_calls.append((len(returns), self.target_episodes))

    try:
        trainer = VectorTrainer(env, seed=42)
        result = trainer.train(
            vector_training_config(
                num_episodes=2,
                adaptive_extension_window=1,
            ),
            plotter=Plotter(),
        )
    finally:
        env.close()

    assert len(result.episode_returns) == 4
    assert any(length >= 2 and target == 4 for length, target in plot_calls)


def test_vector_training_stops_early_when_midpoint_mean_is_too_low(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "dqn.vector_training._early_stopping_score",
        lambda episode_returns, **_kwargs: (
            -300.0 if len(episode_returns) >= 2 else None
        ),
    )
    env = vector_env(num_envs=2)

    try:
        trainer = VectorTrainer(env, seed=42)
        result = trainer.train(
            vector_training_config(
                num_episodes=4,
                adaptive_extension_window=1,
                early_stopping_score=-250.0,
            ),
        )
    finally:
        env.close()

    assert len(result.episode_returns) == 2
    assert result.early_stopped
    assert result.early_stopping_score == -300.0


def test_vector_training_calls_after_episode_hook() -> None:
    env = vector_env(num_envs=2)
    hook_calls = []

    class HookTrainer(VectorTrainer):
        def _after_episode(
            self,
            episode_returns,
            episode_lengths,
            episode_epsilons,
            config,
            plotter=None,
        ) -> None:
            super()._after_episode(
                episode_returns,
                episode_lengths,
                episode_epsilons,
                config,
                plotter,
            )
            hook_calls.append(
                (
                    list(episode_returns),
                    list(episode_lengths),
                    list(episode_epsilons),
                )
            )

    try:
        trainer = HookTrainer(env, seed=42)
        trainer.train(vector_training_config(num_episodes=2))
    finally:
        env.close()

    assert hook_calls
    assert len(hook_calls[-1][0]) == 2
    assert len(hook_calls[-1][1]) == 2
    assert len(hook_calls[-1][2]) == 2


def test_adaptive_training_extension_detects_armstrong_momentum() -> None:
    late_bloomer = [80.0] * 50 + [120.0] * 50
    exhausted = [110.0] * 50 + [112.0] * 50

    assert _should_extend_training(
        late_bloomer,
        window=50,
        base_num_episodes=100,
    )
    assert not _should_extend_training(
        exhausted,
        window=50,
        base_num_episodes=100,
    )
    assert not _should_extend_training(
        late_bloomer * 4,
        window=50,
        base_num_episodes=100,
    )


def test_early_stopping_score_uses_trailing_window_at_midpoint() -> None:
    assert _early_stopping_score(
        [-100.0, -300.0],
        window=2,
        base_num_episodes=4,
        threshold=-250.0,
    ) is None
    assert _early_stopping_score(
        [-300.0, -300.0],
        window=2,
        base_num_episodes=4,
        threshold=-250.0,
    ) == -300.0


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
