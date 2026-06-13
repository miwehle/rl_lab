"""Tuned DQN trainer with a few practical training improvements."""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dateutil import tz
import torch

from dqn.checkpointing import save_checkpoint
from dqn.training import Trainer, TrainingConfig


@dataclass(kw_only=True)
class TuningConfig:
    learning_starts: int = 1000
    optimize_every: int = 4
    double_dqn: bool = False
    save_best_checkpoint: bool = False
    checkpoint_window: int = 50
    checkpoint_min_score: float = 0.0
    checkpoint_min_score_delta: float = 0.0
    checkpoint_path: str | Path = "best_checkpoint.pt"
    log_path: str | Path | None = None

    def __post_init__(self) -> None:
        """Validate config."""
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if self.optimize_every < 1:
            raise ValueError("optimize_every must be >= 1")
        if self.checkpoint_window < 1:
            raise ValueError("checkpoint_window must be >= 1")
        if self.checkpoint_min_score_delta < 0:
            raise ValueError("checkpoint_min_score_delta must be >= 0")


class TunedTrainer(Trainer):
    """DQN trainer with practical tuning features and optional Double DQN targets."""

    def __init__(
        self,
        *args,
        tuning_config: TuningConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tuning_config = tuning_config or TuningConfig()
        self.best_mean_return = float("-inf")
        self.best_checkpoint_score = float("-inf")
        self.checkpoint_returns: list[float] = []

    def _should_optimize(self, config: TrainingConfig) -> bool:
        tuning_config = self.tuning_config
        return (
            len(self.memory) >= config.batch_size
            and self.steps_done >= tuning_config.learning_starts
            and self.steps_done % tuning_config.optimize_every == 0
        )

    def _next_q_values(
        self,
        next_states: tuple[torch.Tensor | None, ...],
        batch_size: int,
    ) -> torch.Tensor:
        if not self.tuning_config.double_dqn:
            return super()._next_q_values(next_states, batch_size)

        next_q_values = torch.zeros(batch_size, device=self.device)
        non_final_mask = torch.tensor(
            tuple(state is not None for state in next_states),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_states = [state for state in next_states if state is not None]

        if non_final_states:
            with torch.no_grad():
                states = torch.cat(non_final_states)
                next_actions = self.q_net(states).max(1).indices.unsqueeze(1)
                next_q_values[non_final_mask] = self.target_net(states).gather(
                    1,
                    next_actions,
                ).squeeze(1)

        return next_q_values

    def _after_episode(
        self,
        episode_returns: list[float],
        episode_lengths: list[int],
        config: TrainingConfig,
        plotter=None,
    ) -> None:
        tuning_config = self.tuning_config
        if plotter is not None and self.steps_done >= tuning_config.learning_starts:
            plotter.mark_episode(len(episode_returns) - 1, "Learning starts")

        super()._after_episode(episode_returns, episode_lengths, config, plotter)

        if tuning_config.log_path is not None:
            self._log_episode(episode_returns, config, tuning_config)

        self.checkpoint_returns.append(episode_returns[-1])

        if not tuning_config.save_best_checkpoint:
            return

        checkpoint_returns = self.checkpoint_returns[-tuning_config.checkpoint_window :]
        checkpoint_score = sum(checkpoint_returns) / len(checkpoint_returns)

        if checkpoint_score < tuning_config.checkpoint_min_score:
            return
        if (
            checkpoint_score
            < self.best_checkpoint_score + tuning_config.checkpoint_min_score_delta
        ):
            return

        self.best_checkpoint_score = checkpoint_score
        save_checkpoint(self, tuning_config.checkpoint_path)
        if plotter is not None:
            plotter.mark_episode(len(episode_returns) - 1, "Checkpoint", repeat=True)

    def _log_episode(
        self,
        episode_returns: list[float],
        config: TrainingConfig,
        tuning_config: TuningConfig,
    ) -> None:
        window_returns = episode_returns[-50:]
        mean_return = sum(window_returns) / len(window_returns)
        self.best_mean_return = max(self.best_mean_return, mean_return)

        log_path = Path(tuning_config.log_path)
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
