"""Robust evaluation for concrete saved checkpoints."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dqn.model import DQN
from dqn.training import ModelFactory, resolve_device
from hpo.checkpointing import load_checkpoint, trial_best_checkpoint_score
from hpo.objective import ObjectiveConfig, evaluate_greedy_q_net
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

    device = resolve_device(objective_cfg.device)
    make_envs = objective_cfg.environment_factory.evaluation_envs()
    q_net_env = next(iter(make_envs.values()))
    candidate_scores: list[list[float]] = [
        [candidate.score] for candidate in candidates
    ]
    results = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        q_net = q_net_from_checkpoint(
            candidate.path,
            make_env=q_net_env,
            device=device,
            model_factory=model_factory,
        )
        world_scores = {
            name: evaluate_greedy_q_net(
                q_net=q_net,
                device=device,
                make_env=make_env,
                episodes=eval_episodes,
                max_steps=objective_cfg.eval_max_steps,
                seed=objective_cfg.eval_seed,
            )
            for name, make_env in make_envs.items()
        }
        robust_score = sum(world_scores.values()) / len(world_scores)
        candidate_scores[candidate_index - 1].append(robust_score)
        mean_score = sum(candidate_scores[candidate_index - 1]) / len(
            candidate_scores[candidate_index - 1]
        )
        result = {
            "trial_number": candidate.trial_number,
            "checkpoint_path": str(candidate.path),
            "source_score": candidate.score,
            "robust_score": robust_score,
            "score": mean_score,
            "world_scores": world_scores,
            "eval_episodes": eval_episodes,
        }
        results.append(result)
        if progress_fn is not None:
            progress_fn(_progress(candidate_index, len(candidates), candidate_scores))

    study.set_user_attr("checkpoint_robustness", results)
    return results


def q_net_from_checkpoint(
    path: str | Path,
    *,
    make_env,
    device=None,
    model_factory: ModelFactory = DQN,
):
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
    )
