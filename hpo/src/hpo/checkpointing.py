"""Checkpoint the best concrete model produced during an HPO trial."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dqn.vector_training import VectorTrainer, VectorTrainingConfig
from hpo.objective import ObjectiveContext
from hpo.study_reporting import TrainingProgressFn, TrainingProgressPlotter


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class ModelCheckpoint:
    path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BestCheckpoint:
    path: Path
    score: float
    episode: int
    window: int | None
    source: str


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


class EvaluationBestCheckpointRecorder:
    """Save q_net whenever evaluation reaches a new best score."""

    def __init__(
        self,
        path: str | Path,
        *,
        min_score: float | None = None,
        min_score_delta: float = 0.0,
    ) -> None:
        if min_score_delta < 0:
            raise ValueError("min_score_delta must be >= 0")
        self.path = Path(path)
        self.min_score = min_score
        self.min_score_delta = min_score_delta
        self.best_score = float("-inf")
        self.best_checkpoint: ModelCheckpoint | None = None

    def after_evaluation(self, ctx: ObjectiveContext) -> None:
        if (
            ctx.q_net is None
            or ctx.score is None
            or ctx.training_result is None
        ):
            return
        if self.min_score is not None and ctx.score < self.min_score:
            return
        if ctx.score < self.best_score + self.min_score_delta:
            return

        metadata = {
            "score": ctx.score,
            "episode": len(ctx.training_result.episode_returns),
            "window": None,
            "trial_number": ctx.trial.number,
            "training_config": asdict(ctx.training_config),
            "world_scores": ctx.world_scores,
        }
        save_checkpoint(ctx.q_net, self.path, metadata)
        self.best_score = ctx.score
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
        episode_env_indices: list[int],
        config: VectorTrainingConfig,
        plotter=None,
    ) -> None:
        self.checkpoint_recorder.after_episode(self, episode_returns)
        super()._after_episode(
            episode_returns,
            episode_lengths,
            episode_epsilons,
            episode_env_indices,
            config,
            plotter,
        )


@dataclass(frozen=True)
class ObjectiveHookFactory:
    checkpoint_dir: str | Path
    window: int = 100
    min_score: float | None = None
    min_score_delta: float = 0.0
    training_progress_fn: TrainingProgressFn | None = None

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.min_score_delta < 0:
            raise ValueError("min_score_delta must be >= 0")

    def with_min_score(self, min_score: float) -> "ObjectiveHookFactory":
        if self.min_score is not None:
            min_score = max(self.min_score, min_score)
        return replace(self, min_score=min_score)

    def with_training_progress(
        self,
        progress_fn: TrainingProgressFn | None,
    ) -> "ObjectiveHookFactory":
        return replace(self, training_progress_fn=progress_fn)

    def for_trial(
        self,
        ctx: ObjectiveContext,
    ) -> "ObjectiveHooks":
        trial = ctx.trial
        checkpoint_subdir = getattr(trial, "checkpoint_subdir", "trials")
        checkpoint_stem = getattr(
            trial,
            "checkpoint_stem",
            f"trial_{trial.number:04d}",
        )
        recorder = BestCheckpointRecorder(
            Path(self.checkpoint_dir)
            / checkpoint_subdir
            / f"{checkpoint_stem}_best.pt",
            window=self.window,
            min_score=self.min_score,
            min_score_delta=self.min_score_delta,
            metadata={
                "trial_number": trial.number,
                "training_config": asdict(ctx.training_config),
            },
        )
        evaluation_recorder = EvaluationBestCheckpointRecorder(
            Path(self.checkpoint_dir)
            / checkpoint_subdir
            / f"{checkpoint_stem}_eval_best.pt",
            min_score=self.min_score,
            min_score_delta=self.min_score_delta,
        )
        return ObjectiveHooks(
            recorder=recorder,
            evaluation_recorder=evaluation_recorder,
            trial_number=trial.number,
            target_episodes=ctx.training_config.num_episodes,
            training_progress_fn=self.training_progress_fn,
        )

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
    evaluation_recorder: EvaluationBestCheckpointRecorder
    trial_number: int
    target_episodes: int
    training_progress_fn: TrainingProgressFn | None = None
    env_labels: list[str | None] | None = None

    @property
    def checkpoint_window(self) -> int:
        return self.recorder.window

    @property
    def checkpoint_min_score(self) -> float | None:
        return self.recorder.min_score

    @property
    def best_checkpoint_score(self) -> float | None:
        if self.recorder.best_checkpoint is None:
            return None
        return self.recorder.best_score

    def training_plotter(self) -> TrainingProgressPlotter | None:
        if self.training_progress_fn is None:
            return None
        return TrainingProgressPlotter(
            trial_number=self.trial_number,
            target_episodes=self.target_episodes,
            progress_fn=self.training_progress_fn,
            checkpoint_window=self.recorder.window,
            checkpoint_min_score=self.recorder.min_score,
            best_checkpoint_score=lambda: self.best_checkpoint_score,
            env_labels=self.env_labels,
        )

    def make_trainer(
        self,
        env,
        *,
        seed: int | None,
        device: Any,
        replay_memory_capacity: int,
    ) -> CheckpointingTrainer:
        self.env_labels = _env_labels(env)
        return CheckpointingTrainer(
            env,
            seed=seed,
            device=device,
            replay_memory_capacity=replay_memory_capacity,
            checkpoint_recorder=self.recorder,
        )

    def q_net_for_evaluation(self, ctx: ObjectiveContext) -> Any:
        q_net = ctx.training_result.q_net
        if self.recorder.best_checkpoint is not None:
            load_checkpoint(
                q_net,
                self.recorder.best_checkpoint.path,
                getattr(ctx.trainer, "device", None),
            )
        return q_net

    def finalize_trial(self, ctx: ObjectiveContext) -> None:
        def save(key, value):
            ctx.trial.set_user_attr(key, value)

        self.evaluation_recorder.after_evaluation(ctx)
        if self.recorder.best_checkpoint is not None:
            metadata = self.recorder.best_checkpoint.metadata
            save("checkpoint_path", str(self.recorder.best_checkpoint.path))
            save("checkpoint_score", metadata["score"])
            save("checkpoint_episode", metadata["episode"])
            save("checkpoint_window", metadata["window"])
        if self.evaluation_recorder.best_checkpoint is not None:
            metadata = self.evaluation_recorder.best_checkpoint.metadata
            save(
                "evaluation_checkpoint_path",
                str(self.evaluation_recorder.best_checkpoint.path),
            )
            save("evaluation_checkpoint_score", metadata["score"])
            save("evaluation_checkpoint_episode", metadata["episode"])


def _env_labels(env: Any) -> list[str | None] | None:
    envs = getattr(env, "envs", None)
    if envs is None:
        return None
    return [
        getattr(getattr(sub_env, "world", None), "name", None)
        for sub_env in envs
    ]


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


def best_checkpoint(study: Any) -> BestCheckpoint:
    """Return the highest-scoring checkpoint saved for a study."""
    checkpoint_dir = Path(study.user_attrs["checkpoint_dir"])
    checkpoints = _checkpoint_references(
        checkpoint_dir,
        pattern="*_eval_best.pt",
    ) or _checkpoint_references(checkpoint_dir, pattern="*_best.pt")
    if not checkpoints:
        raise ValueError("study has no checkpoints")

    return max(checkpoints, key=lambda checkpoint: checkpoint.score)


def _checkpoint_references(
    checkpoint_dir: Path,
    *,
    pattern: str,
) -> list[BestCheckpoint]:
    return [
        _checkpoint_reference(path)
        for subdir in ("trials", "robustness")
        for path in (checkpoint_dir / subdir).glob(pattern)
    ]


def _checkpoint_reference(path: Path) -> BestCheckpoint:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {checkpoint.get('version')}")

    metadata = checkpoint["metadata"]
    return BestCheckpoint(
        path=path,
        score=metadata["score"],
        episode=metadata["episode"],
        window=metadata.get("window"),
        source=path.parent.name,
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
