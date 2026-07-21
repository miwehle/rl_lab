"""Evaluate distilled students in SolarSystemLander."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from dqn.training import resolve_device
from hpo.checkpointing import load_checkpoint
from hpo.environments.solar_system_lander.env import DEFAULT_WORLD_MIX, EnvFactory

from distillation.infra import InfraCfg
from distillation.models import StudentDQN
from distillation.train import StudentRef


def evaluate_student(
    student: StudentRef,
    *,
    eval_episodes_per_world: int = 100,
    eval_seed: int = 0,
    max_steps: int = 1000,
    worlds: tuple[str, ...] | None = None,
    device=None,
    cfg: InfraCfg = InfraCfg(),
) -> dict:
    """Evaluate a student greedily and save an evaluation summary beside the checkpoint."""
    if eval_episodes_per_world < 1:
        raise ValueError("eval_episodes_per_world must be >= 1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    cfg.prepare()
    device = resolve_device(device)
    env_factory = EnvFactory("10d", world_mix=DEFAULT_WORLD_MIX)
    selected_worlds = tuple(worlds or DEFAULT_WORLD_MIX.keys())
    q_net = _load_student(student, env_factory, selected_worlds[0], device=device)

    rows = []
    for world in selected_worlds:
        for episode in range(eval_episodes_per_world):
            seed = eval_seed + episode
            score = _episode_return(q_net, env_factory.make_env(world), device, seed=seed, max_steps=max_steps)
            rows.append({"world": str(world), "episode": episode, "seed": seed, "score": score})

    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    world_scores = {
        world: float(np.mean([row["score"] for row in rows if row["world"] == world]))
        for world in selected_worlds
    }
    summary = {
        "checkpoint_path": str(student.checkpoint_path),
        "eval_episodes_per_world": eval_episodes_per_world,
        "episodes": len(rows),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "q05": float(np.quantile(scores, 0.05)),
        "q25": float(np.quantile(scores, 0.25)),
        "q75": float(np.quantile(scores, 0.75)),
        "q95": float(np.quantile(scores, 0.95)),
        "max": float(np.max(scores)),
        "world_scores": world_scores,
        "teacher_name": student.metadata.get("dataset_metadata", {}).get("teacher_name"),
        "dataset_path": student.metadata.get("dataset_path"),
        "student_hidden_sizes": student.metadata.get("student_hidden_sizes"),
        "rows": rows,
    }
    path = student.checkpoint_path.parent / "evaluation_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_student(student: StudentRef, env_factory: EnvFactory, world: str, *, device) -> StudentDQN:
    env = env_factory.make_env(world)
    try:
        observation, _ = env.reset(seed=0)
        hidden_sizes = tuple(int(value) for value in student.metadata["student_hidden_sizes"])
        q_net = StudentDQN(math.prod(tuple(observation.shape)), env.action_space.n, hidden_sizes).to(device)
        load_checkpoint(q_net, student.checkpoint_path, device)
        q_net.eval()
        return q_net
    finally:
        env.close()


def _episode_return(q_net, env, device, *, seed: int, max_steps: int) -> float:
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    try:
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
