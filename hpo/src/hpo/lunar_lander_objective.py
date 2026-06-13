"""Optuna objective for tuning DQN on LunarLander.

The ``*_from_trial`` functions define the hyperparameter search space. Values
created with ``trial.suggest_*`` are optimized by Optuna; plain constants are
deliberately fixed for all trials.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym

from dqn.training import TrainingConfig
from dqn.tuned_training import TunedTrainer, TuningConfig


def mean_last(values: Sequence[float], window: int) -> float:
    """Return the mean over the last window values."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if not values:
        raise ValueError("values must not be empty")

    tail = values[-window:]
    return sum(tail) / len(tail)


def create_lunar_lander_objective(
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
        training_config = _training_config_from_trial(trial, num_episodes)
        replay_memory_capacity = _replay_memory_capacity_from_trial(trial)
        trial_seed = None if seed is None else seed + trial.number
        tuning_config = _tuning_config_from_trial(
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

        return mean_last(result.episode_returns, score_window)

    return objective


def _training_config_from_trial(trial: Any, num_episodes: int) -> TrainingConfig:
    return TrainingConfig(
        num_episodes=num_episodes,
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        eps_start=0.9,
        eps_end=0.01,
        eps_decay=trial.suggest_int("eps_decay", 10_000, 100_000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        gamma=trial.suggest_float("gamma", 0.97, 0.999),
        tau=trial.suggest_float("tau", 0.001, 0.02, log=True),
    )


def _replay_memory_capacity_from_trial(trial: Any) -> int:
    return 10_000


def _tuning_config_from_trial(
    trial: Any,
    *,
    output_dir: Path | None,
) -> TuningConfig:
    log_path = None
    if output_dir is not None:
        trial_dir = output_dir / f"trial_{trial.number:04d}"
        log_path = trial_dir / "episodes.csv"

    return TuningConfig(
        learning_starts=trial.suggest_int("learning_starts", 1_000, 20_000, log=True),
        optimize_every=trial.suggest_categorical("optimize_every", [1, 2, 4, 8]),
        double_dqn=trial.suggest_categorical("double_dqn", [True, False]),
        save_best_checkpoint=False,
        log_path=log_path,
    )
