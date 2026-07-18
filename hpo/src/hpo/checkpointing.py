"""Checkpoint the best concrete model produced during an HPO trial."""

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dqn.vector_training import VectorTrainer, VectorTrainingConfig
from hpo.objective import ObjectiveContext
from hpo.study.infra_cfg import InfraCfg
from hpo.study.reporting import TrainingProgressFn, TrainingProgressPlotter

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class _ModelCheckpoint:
    path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BestCheckpoint:
    path: Path
    score: float
    episode: int
    window: int | None
    source: str


def trial_best_checkpoint_score(trial: Any) -> float:
    """Score used to carry a trial into the Study plot and robustness selection.

    Prefer the greedy evaluation score of the trial's best checkpoint, falling
    back to the Optuna trial value for older studies or trials without one.
    """
    return float(getattr(trial, "user_attrs", {}).get("evaluation_checkpoint_score", trial.value))


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
        self.best_checkpoint: _ModelCheckpoint | None = None

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
        metadata = self.metadata | {"score": score, "episode": episode, "window": self.window}
        save_checkpoint(trainer.q_net, self.path, metadata)
        self.best_score = score
        self.best_episode = episode
        self.best_checkpoint = _ModelCheckpoint(self.path, metadata)


class EvaluationBestCheckpointRecorder:
    """Save q_net whenever evaluation reaches a new best score."""

    def __init__(
        self, path: str | Path, *, min_score: float | None = None, min_score_delta: float = 0.0
    ) -> None:
        if min_score_delta < 0:
            raise ValueError("min_score_delta must be >= 0")
        self.path = Path(path)
        self.min_score = min_score
        self.min_score_delta = min_score_delta
        self.best_score = float("-inf")
        self.best_checkpoint: _ModelCheckpoint | None = None

    def after_evaluation(self, ctx: ObjectiveContext) -> None:
        if ctx.q_net is None or ctx.score is None or ctx.training_result is None:
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
        self.best_checkpoint = _ModelCheckpoint(self.path, metadata)


class _BestEvalCheckpointArchive:
    """Keep the best evaluation checkpoint in a durable archive directory."""

    checkpoint_name = "best_eval_checkpoint.pt"
    metadata_name = "best_eval_checkpoint.json"

    def __init__(self, archive_dir: str | Path) -> None:
        self.archive_dir = Path(archive_dir)
        self.checkpoint_path = self.archive_dir / self.checkpoint_name
        self.metadata_path = self.archive_dir / self.metadata_name

    def archive(self, checkpoint: _ModelCheckpoint) -> Path | None:
        if checkpoint.metadata["score"] <= self._best_score():
            return None

        self.archive_dir.mkdir(parents=True, exist_ok=True)
        state = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        torch.save(state_dict, self.checkpoint_path)
        self._write_metadata(checkpoint.metadata | {"source_path": str(checkpoint.path)})
        return self.checkpoint_path

    def _best_score(self) -> float:
        if self.metadata_path.exists():
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))["score"]
        return float("-inf")

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        temporary = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.metadata_path)


class _CheckpointingTrainer(VectorTrainer):
    """VectorTrainer variant that records the best model seen during training."""

    def __init__(self, *args, checkpoint_recorder: BestCheckpointRecorder, **kwargs) -> None:
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
            episode_returns, episode_lengths, episode_epsilons, episode_env_indices, config, plotter
        )


@dataclass(frozen=True)
class ObjectiveHookFactory:
    """Create objective hooks for checkpointing one HPO study."""

    study_name: str
    window: int = 100
    min_score: float | None = None
    min_score_delta: float = 0.0
    training_progress_fn: TrainingProgressFn | None = None
    initial_checkpoint_study_name: str | None = None
    cfg: InfraCfg = field(default_factory=InfraCfg)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.min_score_delta < 0:
            raise ValueError("min_score_delta must be >= 0")

    def with_min_score(self, min_score: float) -> "ObjectiveHookFactory":
        if self.min_score is not None:
            min_score = max(self.min_score, min_score)
        return replace(self, min_score=min_score)

    def with_training_progress(self, progress_fn: TrainingProgressFn | None) -> "ObjectiveHookFactory":
        return replace(self, training_progress_fn=progress_fn)

    def for_trial(self, ctx: ObjectiveContext) -> "_ObjectiveHooks":
        trial = ctx.trial
        checkpoint_subdir = getattr(trial, "checkpoint_subdir", "trials")
        checkpoint_stem = getattr(trial, "checkpoint_stem", f"trial_{trial.number:04d}")
        checkpoint_dir = self._checkpoint_dir()
        recorder = BestCheckpointRecorder(
            checkpoint_dir / checkpoint_subdir / f"{checkpoint_stem}_best.pt",
            window=self.window,
            min_score=self.min_score,
            min_score_delta=self.min_score_delta,
            metadata={"trial_number": trial.number, "training_config": asdict(ctx.training_config)},
        )
        evaluation_recorder = EvaluationBestCheckpointRecorder(
            checkpoint_dir / checkpoint_subdir / f"{checkpoint_stem}_eval_best.pt",
            min_score=self.min_score,
            min_score_delta=self.min_score_delta,
        )
        best_eval_archive_dir = self._best_eval_archive_dir()
        archive = None if best_eval_archive_dir is None else _BestEvalCheckpointArchive(best_eval_archive_dir)
        return _ObjectiveHooks(
            recorder=recorder,
            evaluation_recorder=evaluation_recorder,
            best_eval_archive=archive,
            initial_checkpoint_path=self._initial_checkpoint_path(),
            trial_number=trial.number,
            target_episodes=ctx.training_config.num_episodes,
            training_progress_fn=self.training_progress_fn,
            trial_params=ctx.params,
            optimized_param_names=list(getattr(trial, "params", {})),
        )

    def study_attrs(self) -> dict[str, Any]:
        attrs = {
            "checkpoint_dir": str(self._checkpoint_dir()),
            "checkpoint_window": self.window,
            "checkpoint_min_score": self.min_score,
            "checkpoint_min_score_delta": self.min_score_delta,
        }
        best_eval_archive_dir = self._best_eval_archive_dir()
        if best_eval_archive_dir is not None:
            attrs["best_eval_archive_dir"] = str(best_eval_archive_dir)
        initial_checkpoint_path = self._initial_checkpoint_path()
        if initial_checkpoint_path is not None:
            attrs["initial_checkpoint_path"] = str(initial_checkpoint_path)
        return attrs

    def _checkpoint_dir(self) -> Path:
        self.cfg.prepare()
        return self.cfg.checkpoint_dir(self.study_name)

    def _best_eval_archive_dir(self) -> Path:
        self.cfg.prepare()
        return self.cfg.best_eval_archive_dir(self.study_name)

    def _initial_checkpoint_path(self) -> Path | None:
        if self.initial_checkpoint_study_name is None:
            return None
        self.cfg.prepare()
        return self.cfg.best_eval_checkpoint_path(self.initial_checkpoint_study_name)


@dataclass
class _ObjectiveHooks:
    recorder: BestCheckpointRecorder
    evaluation_recorder: EvaluationBestCheckpointRecorder
    best_eval_archive: _BestEvalCheckpointArchive | None
    initial_checkpoint_path: str | Path | None
    trial_number: int
    target_episodes: int
    training_progress_fn: TrainingProgressFn | None = None
    env_labels: list[str | None] | None = None
    trial_params: dict[str, Any] | None = None
    optimized_param_names: list[str] | None = None

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
            trial_params=self.trial_params,
            optimized_param_names=self.optimized_param_names,
        )

    def make_trainer(
        self, env, *, seed: int | None, device: Any, replay_memory_capacity: int, hidden_size: int
    ) -> _CheckpointingTrainer:
        self.env_labels = _env_labels(env)
        trainer = _CheckpointingTrainer(
            env,
            seed=seed,
            device=device,
            replay_memory_capacity=replay_memory_capacity,
            hidden_size=hidden_size,
            checkpoint_recorder=self.recorder,
        )
        if self.initial_checkpoint_path is not None:
            load_checkpoint(trainer.q_net, self.initial_checkpoint_path, trainer.device)
            trainer.target_net.load_state_dict(trainer.q_net.state_dict())
        return trainer

    def q_net_for_evaluation(self, ctx: ObjectiveContext) -> Any:
        q_net = ctx.training_result.q_net
        if self.recorder.best_checkpoint is not None:
            load_checkpoint(q_net, self.recorder.best_checkpoint.path, getattr(ctx.trainer, "device", None))
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
            save("evaluation_checkpoint_path", str(self.evaluation_recorder.best_checkpoint.path))
            save("evaluation_checkpoint_score", metadata["score"])
            save("evaluation_checkpoint_episode", metadata["episode"])
            if self.best_eval_archive is not None:
                archived_path = self.best_eval_archive.archive(self.evaluation_recorder.best_checkpoint)
                if archived_path is not None:
                    save("evaluation_checkpoint_archive_path", str(archived_path))


def _env_labels(env: Any) -> list[str | None] | None:
    envs = getattr(env, "envs", None)
    if envs is None:
        return None
    return [_env_label(sub_env) for sub_env in envs]


def _env_label(env: Any) -> str | None:
    while env is not None:
        world = getattr(env, "world", None)
        if world is not None:
            return getattr(world, "name", None)
        env = getattr(env, "env", None)
    return None


def save_checkpoint(q_net: nn.Module, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q_net.state_dict(), path)
    if metadata is not None:
        _write_checkpoint_metadata(path, metadata)


def checkpoint_metadata_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".json")


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(checkpoint_metadata_path(path).read_text(encoding="utf-8"))


def _write_checkpoint_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    json.dumps(metadata)
    metadata_path = checkpoint_metadata_path(path)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)


def best_checkpoint(study: Any) -> BestCheckpoint:
    """Return the highest-scoring checkpoint saved for a study."""
    checkpoint_dir = Path(study.user_attrs["checkpoint_dir"])
    checkpoints = _checkpoint_references(checkpoint_dir, pattern="*_eval_best.pt") or _checkpoint_references(
        checkpoint_dir, pattern="*_best.pt"
    )
    if not checkpoints:
        raise ValueError("study has no checkpoints")

    return max(checkpoints, key=lambda checkpoint: checkpoint.score)


def _checkpoint_references(checkpoint_dir: Path, *, pattern: str) -> list[BestCheckpoint]:
    return [
        _checkpoint_reference(path)
        for subdir in ("trials", "robustness")
        for path in (checkpoint_dir / subdir).glob(pattern)
    ]


def _checkpoint_reference(path: Path) -> BestCheckpoint:
    metadata = checkpoint_metadata(path)
    return BestCheckpoint(
        path=path,
        score=metadata["score"],
        episode=metadata["episode"],
        window=metadata.get("window"),
        source=path.parent.name,
    )


def load_checkpoint(q_net: nn.Module, path: str | Path, device=None) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    q_net.load_state_dict(state_dict)
