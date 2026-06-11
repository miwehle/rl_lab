"""Tuned DQN trainer with a few practical training improvements."""

from dataclasses import dataclass
import math
from pathlib import Path
import warnings

import torch

from dqn.training import Trainer, TrainingConfig


@dataclass
class TunedTrainingConfig(TrainingConfig):
    learning_starts: int = 1000
    optimize_every: int = 4
    checkpoint_window: int = 10
    checkpoint_path: str | Path = "best_checkpoint.pt"

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if self.optimize_every < 1:
            raise ValueError("optimize_every must be >= 1")
        if self.checkpoint_window < 1:
            raise ValueError("checkpoint_window must be >= 1")

        remaining_exploration = math.exp(-self.learning_starts / self.eps_decay)
        if remaining_exploration < 0.5:
            warnings.warn(
                "eps_decay may be too small relative to learning_starts",
                stacklevel=2,
            )


class TunedTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.best_checkpoint_score = float("-inf")

    def _should_optimize(self, config: TunedTrainingConfig) -> bool:
        return (
            len(self.memory) >= config.batch_size
            and self.steps_done >= config.learning_starts
            and self.steps_done % config.optimize_every == 0
        )

    def _after_episode(
        self,
        episode_returns: list[float],
        episode_lengths: list[int],
        config: TunedTrainingConfig,
        plotter=None,
    ) -> None:
        super()._after_episode(episode_returns, episode_lengths, config, plotter)

        checkpoint_returns = episode_returns[-config.checkpoint_window :]
        checkpoint_score = sum(checkpoint_returns) / len(checkpoint_returns)

        if checkpoint_score <= self.best_checkpoint_score:
            return

        self.best_checkpoint_score = checkpoint_score
        torch.save(self.q_net.state_dict(), config.checkpoint_path)
