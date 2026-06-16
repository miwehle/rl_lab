"""Optuna objective for tuning VectorTrainer DQN on LunarLander."""

from collections.abc import Callable
from time import perf_counter
from typing import Any, Protocol

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch

from dqn.vector_training import VectorTrainer, VectorTrainingConfig
from hpo.evaluation.scoring import best_window_mean


class SearchSpace(Protocol):
    def training_config(self, trial: Any, num_episodes: int) -> VectorTrainingConfig:
        ...

    def replay_memory_capacity(self, trial: Any) -> int:
        ...


VectorEnvFactory = Callable[[str, int], Any]
EnvFactory = Callable[[str], Any]
EvalScoreFn = Callable[..., float]


def create_objective(
    *,
    search_space: SearchSpace,
    num_episodes: int,
    score_window: int = 100,
    seed: int | None = 42,
    env_id: str = "LunarLander-v3",
    device=None,
    num_envs: int = 16,
    eval_episodes: int = 3,
    eval_max_steps: int = 2_000,
    env_factory: EnvFactory = gym.make,
    vector_env_factory: VectorEnvFactory | None = None,
    trainer_factory: type[VectorTrainer] = VectorTrainer,
    eval_score_fn: EvalScoreFn | None = None,
) -> Callable[[Any], float]:
    """Create an Optuna objective that trains one vectorized LunarLander DQN."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    if score_window < 1:
        raise ValueError("score_window must be >= 1")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be >= 1")
    if eval_max_steps < 1:
        raise ValueError("eval_max_steps must be >= 1")

    if vector_env_factory is None:
        def make_vector_env(env_id: str, num_envs: int) -> Any:
            return _make_vector_env(env_id, num_envs, env_factory)
    else:
        make_vector_env = vector_env_factory

    evaluate = eval_score_fn or evaluate_greedy_policy

    def objective(trial: Any) -> float:
        training_config = search_space.training_config(trial, num_episodes)
        replay_memory_capacity = search_space.replay_memory_capacity(trial)
        trial_seed = None if seed is None else seed + trial.number

        env = make_vector_env(env_id, num_envs)
        try:
            trainer = trainer_factory(
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

        best_window = best_window_mean(result.episode_returns, score_window)
        final_returns = result.episode_returns[-score_window:]
        final_window_score = sum(final_returns) / len(final_returns)
        objective_score = (best_window.mean + final_window_score) / 2
        eval_score = evaluate(
            q_net=result.q_net,
            device=trainer.device,
            env_id=env_id,
            env_factory=env_factory,
            episodes=eval_episodes,
            max_steps=eval_max_steps,
            seed=trial_seed,
        )

        trial.set_user_attr("best_window_score", best_window.mean)
        trial.set_user_attr("best_window_start_episode", best_window.start_episode)
        trial.set_user_attr("best_window_end_episode", best_window.end_episode)
        trial.set_user_attr("final_window_score", final_window_score)
        trial.set_user_attr("objective_score", objective_score)
        trial.set_user_attr("eval_score", eval_score)
        trial.set_user_attr("wall_time_seconds", wall_time_seconds)
        return objective_score

    return objective


def evaluate_greedy_policy(
    *,
    q_net: Any,
    device,
    env_id: str,
    env_factory: EnvFactory = gym.make,
    episodes: int = 3,
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
    env_factory: EnvFactory,
) -> SyncVectorEnv:
    return SyncVectorEnv([lambda: env_factory(env_id) for _ in range(num_envs)])
