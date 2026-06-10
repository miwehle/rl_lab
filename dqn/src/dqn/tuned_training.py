"""Tuned DQN trainer with a few practical training improvements."""

from dataclasses import dataclass
from pathlib import Path

import torch

from dqn.config import TrainingConfig
from dqn.training import Trainer


@dataclass
class TunedTrainingConfig(TrainingConfig):
    learning_starts: int = 1000
    optimize_every: int = 4
    checkpoint_path: str | Path = "best_checkpoint.pt"


class TunedTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.best_checkpoint_return = float("-inf")

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
    ) -> None:
        episode_return = episode_returns[-1]
        if episode_return <= self.best_checkpoint_return:
            return

        self.best_checkpoint_return = episode_return
        torch.save(self.q_net.state_dict(), config.checkpoint_path)
