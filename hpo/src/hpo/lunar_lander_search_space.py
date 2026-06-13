"""Optuna search space for tuning DQN on LunarLander.

Functions in this module define which hyperparameters are optimized. Values
created with ``trial.suggest_*`` are optimized by Optuna; plain constants are
deliberately fixed for all trials.
"""

from pathlib import Path
from typing import Any

from dqn.training import TrainingConfig
from dqn.tuned_training import TuningConfig


def training_config(trial: Any, num_episodes: int) -> TrainingConfig:
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


def replay_memory_capacity(trial: Any) -> int:
    return 10_000


def tuning_config(
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
