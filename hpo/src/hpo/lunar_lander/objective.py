"""Optuna objective for tuning VectorTrainer DQN on LunarLander."""

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any, Protocol

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch

from dqn.vector_training import VectorTrainer, VectorTrainingConfig
from hpo.evaluation.scoring import training_effort
from hpo.lunar_lander.logging import log_call


logger = logging.getLogger(__name__)


class SearchSpace(Protocol):
    def training_config(self, trial: Any, num_episodes: int) -> VectorTrainingConfig:
        ...

    def replay_memory_capacity(self, trial: Any) -> int:
        ...


EnvFactory = Callable[[str], Any]


def create_objective(
    *,
    search_space: SearchSpace,
    num_episodes: int,
    baseline_env_steps: float | None = None,
    baseline_processed_samples: float | None = None,
    alpha: float = 0.5,
    quality_weight: float = 0.9,
    quality_min: float = 200.0,
    quality_target: float = 250.0,
    seed: int | None = 42,
    env_id: str = "LunarLander-v3",
    device=None,
    num_envs: int = 16,
    eval_episodes: int = 20,
    eval_seed: int | None = 10_000,
    eval_max_steps: int = 2_000,
) -> Callable[[Any], float]:
    """Create an Optuna objective that trains one vectorized LunarLander DQN."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be >= 1")
    if eval_max_steps < 1:
        raise ValueError("eval_max_steps must be >= 1")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 <= quality_weight <= 1:
        raise ValueError("quality_weight must be between 0 and 1")
    if quality_target <= quality_min:
        raise ValueError("quality_target must be greater than quality_min")
    if (baseline_env_steps is None) != (baseline_processed_samples is None):
        raise ValueError("baseline effort values must both be set or both be None")

    @log_call
    def objective(trial: Any) -> float:
        training_config = search_space.training_config(trial, num_episodes)
        replay_memory_capacity = search_space.replay_memory_capacity(trial)
        trial_seed = None if seed is None else seed + trial.number

        env = _make_vector_env(env_id, num_envs)
        try:
            trainer = VectorTrainer(
                env,
                seed=trial_seed,
                device=device,
                replay_memory_capacity=replay_memory_capacity,
            )

            start_time = perf_counter()
            result = trainer.train(training_config)
            wall_time_seconds = perf_counter() - start_time
        finally:
            env.close()

        gym_score = evaluate_greedy_policy(
            q_net=result.q_net,
            device=trainer.device,
            env_id=env_id,
            episodes=eval_episodes,
            max_steps=eval_max_steps,
            seed=eval_seed,
        )
        processed_samples = result.optimizer_updates * training_config.batch_size
        effort = (
            1.0
            if baseline_env_steps is None
            else training_effort(
                env_steps=result.env_steps,
                processed_samples=processed_samples,
                baseline_env_steps=baseline_env_steps,
                baseline_processed_samples=baseline_processed_samples,
                alpha=alpha,
            )
        )
        quality = (gym_score - quality_min) / (quality_target - quality_min)
        objective_score = (
            quality_weight * quality
            - (1 - quality_weight) * (effort - 1)
        )

        def save(key, value):
            trial.set_user_attr(key, value)

        save("gym_score", gym_score)
        save("env_steps", result.env_steps)
        save("optimizer_updates", result.optimizer_updates)
        save("processed_samples", processed_samples)
        save("training_effort", effort)
        save("trial_seed", trial_seed)
        save("wall_time_seconds", wall_time_seconds)
        save("training_curve", {
            "episode_returns": result.episode_returns,
            "episode_epsilons": result.episode_epsilons,
        })

        return objective_score

    return objective


@log_call
def evaluate_greedy_policy(
    *,
    q_net: Any,
    device,
    env_id: str,
    env_factory: EnvFactory = gym.make,
    episodes: int = 20,
    max_steps: int = 2_000,
    seed: int | None = None,
) -> float:
    """Return mean greedy episode return for a trained Q-network."""
    episode_returns = []
    q_net.eval()

    for episode in range(episodes):
        env = env_factory(env_id)
        try:
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)
            episode_return = 0.0

            for _ in range(max_steps):
                state = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    action = q_net(state).argmax(dim=1).item()

                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
        finally:
            env.close()

        episode_returns.append(episode_return)

    return sum(episode_returns) / len(episode_returns)


def _make_vector_env(
    env_id: str,
    num_envs: int,
) -> SyncVectorEnv:
    return SyncVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)])
