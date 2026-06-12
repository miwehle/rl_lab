"""Tuned DQN trainer with a few practical training improvements."""

import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import warnings

from dateutil import tz

from dqn.checkpointing import save_checkpoint
from dqn.training import Trainer, TrainingConfig


@dataclass
class TunedTrainingConfig(TrainingConfig):
    log_path: str | Path | None = None
    learning_starts: int = 1000
    optimize_every: int = 4
    save_best_checkpoint: bool = False
    checkpoint_window: int = 10
    checkpoint_min_score: float = 150.0
    checkpoint_min_score_delta: float = 5.0
    checkpoint_path: str | Path = "best_checkpoint.pt"

    def __post_init__(self) -> None:
        """Validate config."""
        super().__post_init__()

        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if self.optimize_every < 1:
            raise ValueError("optimize_every must be >= 1")
        if self.checkpoint_window < 1:
            raise ValueError("checkpoint_window must be >= 1")
        if self.checkpoint_min_score_delta < 0:
            raise ValueError("checkpoint_min_score_delta must be >= 0")

        remaining_exploration = math.exp(-self.learning_starts / self.eps_decay)
        if remaining_exploration < 0.5:
            warnings.warn(
                "eps_decay may be too small relative to learning_starts",
                stacklevel=2,
            )


class TunedTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.best_mean_return = float("-inf")
        self.best_checkpoint_score = float("-inf")
        self.checkpoint_returns: list[float] = []

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
        if plotter is not None and self.steps_done >= config.learning_starts:
            plotter.mark_episode(len(episode_returns) - 1, "Learning starts")

        super()._after_episode(episode_returns, episode_lengths, config, plotter)

        if config.log_path is not None:
            self._log_episode(episode_returns, config)

        self.checkpoint_returns.append(episode_returns[-1])

        if not config.save_best_checkpoint:
            return

        checkpoint_returns = self.checkpoint_returns[-config.checkpoint_window :]
        checkpoint_score = sum(checkpoint_returns) / len(checkpoint_returns)

        if checkpoint_score < config.checkpoint_min_score:
            return
        if checkpoint_score < self.best_checkpoint_score + config.checkpoint_min_score_delta:
            return

        self.best_checkpoint_score = checkpoint_score
        save_checkpoint(self, config.checkpoint_path)
        if plotter is not None:
            plotter.mark_episode(len(episode_returns) - 1, "Checkpoint", repeat=True)

    def _log_episode(
        self,
        episode_returns: list[float],
        config: TunedTrainingConfig,
    ) -> None:
        window_returns = episode_returns[-50:]
        mean_return = sum(window_returns) / len(window_returns)
        self.best_mean_return = max(self.best_mean_return, mean_return)

        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists() or log_path.stat().st_size == 0

        with log_path.open("a", encoding="utf-8", newline="") as log_file:
            writer = csv.DictWriter(
                log_file,
                delimiter=";",
                fieldnames=[
                    "timestamp",
                    "episode",
                    "steps_done",
                    "return",
                    "mean_return",
                    "best_mean_return",
                    "epsilon",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now(tz.gettz("Europe/Berlin")).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "episode": len(episode_returns),
                    "steps_done": self.steps_done,
                    "return": f"{episode_returns[-1]:.1f}".replace(".", ","),
                    "mean_return": f"{mean_return:.1f}".replace(".", ","),
                    "best_mean_return": f"{self.best_mean_return:.1f}".replace(".", ","),
                    "epsilon": f"{self._exploration_rate(config):.3f}".replace(".", ","),
                }
            )
