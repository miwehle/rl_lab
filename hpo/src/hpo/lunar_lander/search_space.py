"""Optuna search space for tuning DQN on LunarLander.

Functions in this module define which hyperparameters are optimized. Values
created with ``trial.suggest_*`` are optimized by Optuna; plain constants are
deliberately fixed for all trials.
"""

from pathlib import Path
from typing import Any

from dqn.training import TrainingConfig
from dqn.tuned_training import TuningConfig


def replay_memory_capacity(trial: Any) -> int:
    return 50_000


def training_config(trial: Any, num_episodes: int) -> TrainingConfig:
    return TrainingConfig(
        num_episodes=num_episodes,
        batch_size=128,
        gamma=0.99,
        eps_start=trial.suggest_float("eps_start", 0.5, 1.0),  # current: 1.0
        eps_end=0.01,
        eps_decay=trial.suggest_int("eps_decay", 10_000, 50_000, log=True),  # current: 20_000
        tau=0.005,
        learning_rate=3e-4,
    )


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
        learning_starts=5_000,
        optimize_every=4,
        double_dqn=False,
        save_best_checkpoint=False,
        checkpoint_min_score=0.0,
        checkpoint_min_score_delta=0.0,
        log_path=log_path,
    )
