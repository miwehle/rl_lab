"""Collect and load teacher datasets for DQN distillation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from dqn.model import DQN
from dqn.training import resolve_device
from hpo.checkpointing import checkpoint_metadata, load_checkpoint
from hpo.environments.solar_system_lander.env import DEFAULT_WORLD_MIX, EnvFactory

from distillation.infra_cfg import InfraCfg

DEFAULT_TEACHER_NAME = "solar_system_lander_10d_elise_stp"
DEFAULT_SEEDS = tuple(range(100))


@dataclass(frozen=True)
class DatasetRef:
    path: Path
    metadata: dict


def collect_teacher_dataset(
    *,
    teacher_name: str = DEFAULT_TEACHER_NAME,
    epsilon: float = 0.05,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    worlds: Sequence[str] | None = None,
    max_steps: int = 1000,
    dataset_name: str | None = None,
    device=None,
    cfg: InfraCfg = InfraCfg(),
) -> DatasetRef:
    """Collect observations and teacher Q-values from teacher rollouts."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if not seeds:
        raise ValueError("seeds must not be empty")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    cfg.prepare()
    device = resolve_device(device)
    env_factory = EnvFactory("10d", world_mix=DEFAULT_WORLD_MIX)
    selected_worlds = tuple(str(world) for world in (worlds or DEFAULT_WORLD_MIX.keys()))
    teacher_path = cfg.teacher_checkpoint_path(teacher_name)
    teacher_metadata = checkpoint_metadata(teacher_path)
    teacher = _load_teacher(teacher_path, env_factory, selected_worlds[0], device=device, metadata=teacher_metadata)
    rng = np.random.default_rng(0)

    observations = []
    teacher_q_values = []
    teacher_actions = []
    rollout_actions = []
    world_labels = []
    seed_values = []
    step_values = []
    scenario_labels = []
    episode_rows = []

    for world in selected_worlds:
        for seed in seeds:
            episode_return = _collect_episode(
                teacher,
                env_factory.make_env(world),
                int(seed),
                world,
                epsilon,
                max_steps,
                rng,
                device,
                observations,
                teacher_q_values,
                teacher_actions,
                rollout_actions,
                world_labels,
                seed_values,
                step_values,
                scenario_labels,
            )
            episode_rows.append({"world": world, "seed": int(seed), **episode_return})

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_name": teacher_name,
        "teacher_checkpoint_path": str(teacher_path),
        "teacher_score": teacher_metadata.get("score"),
        "epsilon": epsilon,
        "seeds": [int(seed) for seed in seeds],
        "worlds": list(selected_worlds),
        "max_steps": max_steps,
        "frames": len(observations),
        "episodes": len(episode_rows),
        "episode_rows": episode_rows,
    }
    path = cfg.dataset_path(dataset_name or _dataset_name(teacher_name, epsilon, seeds, selected_worlds))
    save_dataset(
        path,
        metadata=metadata,
        observations=np.asarray(observations, dtype=np.float32),
        teacher_q_values=np.asarray(teacher_q_values, dtype=np.float32),
        teacher_actions=np.asarray(teacher_actions, dtype=np.int64),
        rollout_actions=np.asarray(rollout_actions, dtype=np.int64),
        worlds=np.asarray(world_labels),
        seeds=np.asarray(seed_values, dtype=np.int64),
        steps=np.asarray(step_values, dtype=np.int64),
        scenarios=np.asarray(scenario_labels),
    )
    return DatasetRef(path=path, metadata=metadata)


def load_dataset(path: str | Path) -> DatasetRef:
    path = Path(path)
    metadata = _metadata_path(path).read_text(encoding="utf-8")
    return DatasetRef(path=path, metadata=json.loads(metadata))


def save_dataset(path: str | Path, *, metadata: dict, **arrays) -> DatasetRef:
    """Save a distillation dataset and its sidecar metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    _metadata_path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DatasetRef(path=path, metadata=metadata)


def dataset_arrays(dataset: DatasetRef) -> dict[str, np.ndarray]:
    with np.load(dataset.path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_teacher(path: Path, env_factory: EnvFactory, world: str, *, device, metadata: dict) -> DQN:
    env = env_factory.make_env(world)
    try:
        observation, _ = env.reset(seed=0)
        hidden_size = int(metadata.get("training_config", {}).get("hidden_size", 128))
        q_net = DQN(len(observation), env.action_space.n, hidden_size).to(device)
        load_checkpoint(q_net, path, device)
        q_net.eval()
        return q_net
    finally:
        env.close()


def _collect_episode(
    teacher,
    env,
    seed: int,
    world: str,
    epsilon: float,
    max_steps: int,
    rng: np.random.Generator,
    device,
    observations,
    teacher_q_values,
    teacher_actions,
    rollout_actions,
    world_labels,
    seed_values,
    step_values,
    scenario_labels,
) -> dict[str, float | int]:
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    try:
        for step in range(max_steps):
            state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = teacher(state)[0].detach().cpu().numpy()
            teacher_action = int(np.argmax(q_values))
            rollout_action = (
                int(env.action_space.sample()) if epsilon > 0.0 and rng.random() < epsilon else teacher_action
            )

            observations.append(np.asarray(observation, dtype=np.float32))
            teacher_q_values.append(q_values.astype(np.float32))
            teacher_actions.append(teacher_action)
            rollout_actions.append(rollout_action)
            world_labels.append(world)
            seed_values.append(seed)
            step_values.append(step)
            scenario_labels.append("epsilon" if rollout_action != teacher_action else "greedy")

            observation, reward, terminated, truncated, _ = env.step(rollout_action)
            episode_return += float(reward)
            if terminated or truncated:
                return {"steps": step + 1, "score": episode_return}
        return {"steps": max_steps, "score": episode_return}
    finally:
        env.close()


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _dataset_name(teacher_name: str, epsilon: float, seeds: Sequence[int], worlds: Sequence[str]) -> str:
    seed_part = f"{min(seeds)}-{max(seeds)}-n{len(seeds)}"
    world_part = "-".join(worlds)
    epsilon_part = str(epsilon).replace(".", "p")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{teacher_name}_eps{epsilon_part}_{world_part}_{seed_part}_{timestamp}"
