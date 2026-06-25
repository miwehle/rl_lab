"""Checkpoint the best concrete model produced during an HPO trial."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dqn.vector_training import VectorTrainer, VectorTrainingConfig


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class ModelCheckpoint:
    path: Path
    metadata: dict[str, Any]


class BestCheckpointRecorder:
    """Save q_net whenever the trailing return window reaches a new best score."""

    def __init__(
        self,
        path: str | Path,
        *,
        window: int,
        min_score: float | None = None,
        min_score_delta: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if min_score_delta < 0:
            raise ValueError("min_score_delta must be >= 0")

        self.path = Path(path)
        self.window = window
        self.min_score = min_score
        self.min_score_delta = min_score_delta
        self.metadata = metadata or {}
        self.best_score = float("-inf")
        self.best_episode: int | None = None
        self.best_checkpoint: ModelCheckpoint | None = None

    def after_episode(self, trainer: VectorTrainer, episode_returns: list[float]) -> None:
        if len(episode_returns) < self.window:
            return

        window_returns = episode_returns[-self.window :]
        score = sum(window_returns) / len(window_returns)
        if self.min_score is not None and score < self.min_score:
            return
        if score < self.best_score + self.min_score_delta:
            return

        episode = len(episode_returns)
        metadata = self.metadata | {
            "score": score,
            "episode": episode,
            "window": self.window,
        }
        save_checkpoint(trainer.q_net, self.path, metadata)
        self.best_score = score
        self.best_episode = episode
        self.best_checkpoint = ModelCheckpoint(self.path, metadata)


class CheckpointingTrainer(VectorTrainer):
    """VectorTrainer variant that records the best model seen during training."""

    def __init__(
        self,
        *args,
        checkpoint_recorder: BestCheckpointRecorder,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint_recorder = checkpoint_recorder

    def _after_episode(
        self,
        episode_returns: list[float],
        episode_lengths: list[int],
        episode_epsilons: list[float],
        config: VectorTrainingConfig,
        plotter=None,
    ) -> None:
        super()._after_episode(
            episode_returns,
            episode_lengths,
            episode_epsilons,
            config,
            plotter,
        )
        self.checkpoint_recorder.after_episode(self, episode_returns)


@dataclass(frozen=True)
class ObjectiveHookFactory:
    checkpoint_dir: str | Path
    window: int = 100
    min_score: float | None = None
    min_score_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.min_score_delta < 0:
            raise ValueError("min_score_delta must be >= 0")

    def for_trial(
        self,
        trial: Any,
        training_config: VectorTrainingConfig,
    ) -> "ObjectiveHooks":
        recorder = BestCheckpointRecorder(
            Path(self.checkpoint_dir) / f"trial_{trial.number:04d}_best.pt",
            window=self.window,
            min_score=self.min_score,
            min_score_delta=self.min_score_delta,
            metadata={
                "trial_number": trial.number,
                "training_config": asdict(training_config),
            },
        )
        return ObjectiveHooks(recorder)

    def study_attrs(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_window": self.window,
            "checkpoint_min_score": self.min_score,
            "checkpoint_min_score_delta": self.min_score_delta,
        }


@dataclass
class ObjectiveHooks:
    recorder: BestCheckpointRecorder

    def make_trainer(
        self,
        env,
        *,
        seed: int | None,
        device: Any,
        replay_memory_capacity: int,
    ) -> CheckpointingTrainer:
        return CheckpointingTrainer(
            env,
            seed=seed,
            device=device,
            replay_memory_capacity=replay_memory_capacity,
            checkpoint_recorder=self.recorder,
        )

    def q_net_for_evaluation(self, q_net: Any, device: Any) -> Any:
        if self.recorder.best_checkpoint is not None:
            load_checkpoint(q_net, self.recorder.best_checkpoint.path, device)
        return q_net

    def save_trial_attrs(self, save: Callable[[str, Any], None]) -> None:
        if self.recorder.best_checkpoint is None:
            return

        metadata = self.recorder.best_checkpoint.metadata
        save("checkpoint_path", str(self.recorder.best_checkpoint.path))
        save("checkpoint_score", metadata["score"])
        save("checkpoint_episode", metadata["episode"])
        save("checkpoint_window", metadata["window"])


def save_checkpoint(
    q_net: nn.Module,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model_state_dict": q_net.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(
    q_net: nn.Module,
    path: str | Path,
    device=None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {checkpoint.get('version')}")

    q_net.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint["metadata"]
