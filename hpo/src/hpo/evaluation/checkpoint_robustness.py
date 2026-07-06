"""Robust evaluation for concrete saved checkpoints."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from dqn.model import DQN
from dqn.training import ModelFactory, resolve_device
from hpo.checkpointing import load_checkpoint, trial_best_checkpoint_score
from hpo.objective import ObjectiveConfig
from hpo.study_reporting import RobustnessProgress


@dataclass(frozen=True)
class CheckpointCandidate:
    trial_number: int
    path: Path
    score: float


def evaluate_checkpoint_robustness(
    *,
    study: Any,
    objective_cfg: ObjectiveConfig,
    top_n: int = 5,
    eval_episodes: int = 20,
    progress_fn=None,
    model_factory: ModelFactory = DQN,
) -> list[dict[str, Any]]:
    """Evaluate top saved eval checkpoints and store their robust scores."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be >= 1")

    candidates = _top_checkpoint_candidates(study, top_n)
    if not candidates:
        raise ValueError("study has no evaluation checkpoints")

    candidate_scores: list[list[float]] = [[candidate.score] for candidate in candidates]
    checkpoint_summaries: list[dict[str, Any]] = []
    results = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        summary = robustness_over_all_worlds(
            candidate.path, objective_cfg, episodes=eval_episodes, progress=False, model_factory=model_factory
        )
        summary = {
            "candidate": candidate_index,
            "trial_number": candidate.trial_number,
            "source_score": candidate.score,
            **summary,
        }
        checkpoint_summaries.append(summary)
        robust_score = summary["mean"]
        candidate_scores[candidate_index - 1].append(robust_score)
        result = {
            "trial_number": candidate.trial_number,
            "checkpoint_path": str(candidate.path),
            "source_score": candidate.score,
            "robust_score": robust_score,
            "score": robust_score,
            "world_scores": summary["world_scores"],
            "checkpoint_summary": summary,
            "eval_episodes": eval_episodes,
        }
        results.append(result)
        if progress_fn is not None:
            progress_fn(_progress(candidate_index, len(candidates), candidate_scores, checkpoint_summaries))

    study.set_user_attr("checkpoint_robustness", results)
    return results


def checkpoint_scores(
    checkpoint_path: str | Path,
    objective_cfg: ObjectiveConfig,
    *,
    episodes: int = 100,
    progress: bool = True,
    model_factory: ModelFactory = DQN,
) -> pd.DataFrame:
    """Return greedy episode scores for each evaluation world."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    device = resolve_device(objective_cfg.device)
    make_envs = objective_cfg.environment_factory.evaluation_envs()
    q_net_env = next(iter(make_envs.values()))
    q_net = q_net_from_checkpoint(
        checkpoint_path, make_env=q_net_env, device=device, model_factory=model_factory
    )
    q_net.eval()

    rows = []
    work_items = (
        (world, make_env, episode) for world, make_env in make_envs.items() for episode in range(episodes)
    )
    for world, make_env, episode in _with_progress(
        work_items, enabled=progress, total=len(make_envs) * episodes, desc="Evaluating checkpoint"
    ):
        seed = None if objective_cfg.eval_seed is None else objective_cfg.eval_seed + episode
        rows.append(
            {
                "world": world,
                "episode": episode,
                "score": _episode_return(
                    q_net, make_env, device, max_steps=objective_cfg.eval_max_steps, seed=seed
                ),
            }
        )

    return pd.DataFrame(rows)


def robustness_over_all_worlds(
    checkpoint_path: str | Path,
    objective_cfg: ObjectiveConfig,
    *,
    episodes: int = 100,
    progress: bool = True,
    model_factory: ModelFactory = DQN,
) -> dict[str, Any]:
    """Return one checkpoint's robustness summary over all evaluation worlds."""
    scores = checkpoint_scores(
        checkpoint_path, objective_cfg, episodes=episodes, progress=progress, model_factory=model_factory
    )
    values = scores["score"]
    world_scores = scores.groupby("world", sort=False)["score"].mean()
    return {
        "checkpoint_path": str(checkpoint_path),
        "episodes_per_world": episodes,
        "episodes": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "q05": float(values.quantile(0.05)),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "q95": float(values.quantile(0.95)),
        "max": float(values.max()),
        "world_scores": {str(world): float(score) for world, score in world_scores.items()},
    }


def score_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Return the checkpoint score summary used by the notebook plots."""
    return scores.groupby("world", sort=False)["score"].agg(
        episodes="count",
        mean="mean",
        std="std",
        min="min",
        q05=lambda score: score.quantile(0.05),
        q25=lambda score: score.quantile(0.25),
        median="median",
        q75=lambda score: score.quantile(0.75),
        q95=lambda score: score.quantile(0.95),
        max="max",
    )


def q_net_from_checkpoint(path: str | Path, *, make_env, device=None, model_factory: ModelFactory = DQN):
    """Build a Q-net from env dimensions and load checkpoint weights into it."""
    device = resolve_device(device)
    hidden_size = _checkpoint_hidden_size(path)
    env = make_env()
    try:
        observation, _ = env.reset()
        n_observations = math.prod(tuple(observation.shape))
        n_actions = env.action_space.n
        q_net = (
            DQN(n_observations, n_actions, hidden_size)
            if model_factory is DQN
            else model_factory(n_observations, n_actions)
        ).to(device)
        load_checkpoint(q_net, path, device)
        return q_net
    finally:
        env.close()


def _episode_return(q_net, make_env, device, *, max_steps: int, seed: int | None) -> float:
    env = make_env()
    try:
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0

        for _ in range(max_steps):
            state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(q_net(state).argmax(dim=1).item())

            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break

        return episode_return
    finally:
        env.close()


def _with_progress(items, *, enabled: bool, total: int, desc: str):
    if not enabled:
        return items
    tqdm = _tqdm()
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc)


def _tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm


def _checkpoint_hidden_size(path: str | Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    training_config = metadata.get("training_config", {})
    return int(training_config.get("hidden_size", 128))


def _top_checkpoint_candidates(study: Any, top_n: int) -> list[CheckpointCandidate]:
    candidates = [
        CheckpointCandidate(
            trial_number=trial.number,
            path=Path(trial.user_attrs["evaluation_checkpoint_path"]),
            score=trial_best_checkpoint_score(trial),
        )
        for trial in study.trials
        if trial.state.name == "COMPLETE"
        and trial.value is not None
        and "evaluation_checkpoint_path" in getattr(trial, "user_attrs", {})
    ]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:top_n]


def _progress(
    candidate_index: int,
    candidate_count: int,
    candidate_scores: list[list[float]],
    checkpoint_summaries: list[dict[str, Any]] | None = None,
) -> RobustnessProgress:
    return RobustnessProgress(
        candidate_index=candidate_index,
        candidate_count=candidate_count,
        seed_index=1,
        seed_count=1,
        candidate_seed_scores=[list(scores) for scores in candidate_scores],
        title="Checkpoint Robustness Evaluation",
        step_label="Eval",
        first_score_label="Source score",
        extra_score_label="Robust eval",
        checkpoint_summaries=checkpoint_summaries,
    )
