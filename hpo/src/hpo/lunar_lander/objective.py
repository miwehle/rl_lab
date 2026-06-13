"""Optuna objective for tuning DQN on LunarLander."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym

from dqn.tuned_training import TunedTrainer
from hpo.evaluation.scoring import best_window_mean
from hpo.lunar_lander import search_space


def create_objective(
    *,
    num_episodes: int,
    score_window: int = 50,
    seed: int | None = 42,
    output_dir: str | Path | None = None,
    env_id: str = "LunarLander-v3",
    device=None,
    env_factory: Callable[[str], Any] = gym.make,
    trainer_factory: type[TunedTrainer] = TunedTrainer,
) -> Callable[[Any], float]:
    """Create an Optuna objective that trains one LunarLander DQN per trial."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    if score_window < 1:
        raise ValueError("score_window must be >= 1")

    base_output_dir = Path(output_dir) if output_dir is not None else None

    def objective(trial: Any) -> float:
        training_config = search_space.training_config(trial, num_episodes)
        replay_memory_capacity = search_space.replay_memory_capacity(trial)
        trial_seed = None if seed is None else seed + trial.number
        tuning_config = search_space.tuning_config(
            trial,
            output_dir=base_output_dir,
        )

        env = env_factory(env_id)
        try:
            trainer = trainer_factory(
                env,
                seed=trial_seed,
                device=device,
                replay_memory_capacity=replay_memory_capacity,
                tuning_config=tuning_config,
            )
            result = trainer.train(training_config)
        finally:
            env.close()

        score = best_window_mean(result.episode_returns, score_window)
        trial.set_user_attr("best_window_mean", score.mean)
        trial.set_user_attr("best_window_start_episode", score.start_episode)
        trial.set_user_attr("best_window_end_episode", score.end_episode)
        return score.mean

    return objective
