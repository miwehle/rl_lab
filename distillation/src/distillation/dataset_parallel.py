"""Parallel teacher dataset collection for DQN distillation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
import json

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from tqdm.auto import tqdm

from dqn.training import resolve_device
from hpo.checkpointing import checkpoint_metadata
from hpo.environments.solar_system_lander.env import DEFAULT_WORLD_MIX, EnvFactory, EnvWrapper, WORLDS

from distillation.clock import get_clock, lap, reset_clocks, stop, total_lap_times, total_time
from distillation.dataset import (
    DEFAULT_SEEDS,
    DEFAULT_TEACHER_NAME,
    DatasetRef,
    _dataset_name,
    _load_teacher,
    _worlds_from_mix,
    save_dataset,
)
from distillation.infra_cfg import InfraCfg


def collect_teacher_dataset_parallel(
    *,
    teacher_name: str = DEFAULT_TEACHER_NAME,
    epsilon: float = 0.05,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    world_mix: Mapping[str, int] = DEFAULT_WORLD_MIX,
    max_steps: int = 1000,
    dataset_name: str | None = None,
    num_envs: int = 16,
    device=None,
    cfg: InfraCfg = InfraCfg(),
    progress: bool = True,
) -> DatasetRef:
    """Collect observations and teacher Q-values with AsyncVectorEnv batches."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if not seeds:
        raise ValueError("seeds must not be empty")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    reset_clocks()
    collect_clock = get_clock("collect_teacher_dataset_parallel")
    cfg.prepare()
    device = resolve_device(device)
    selected_worlds = _worlds_from_mix(world_mix)
    env_factory = EnvFactory("10d", world_mix=dict.fromkeys(selected_worlds, 1))
    teacher_path = cfg.teacher_checkpoint_path(teacher_name)
    teacher_metadata = checkpoint_metadata(teacher_path)
    lap(collect_clock, "setup")
    teacher = _load_teacher(teacher_path, env_factory, selected_worlds[0], device=device, metadata=teacher_metadata)
    lap(collect_clock, "load_teacher")
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

    episodes = [(world, int(seed)) for world in selected_worlds for seed in seeds]
    batches = [episodes[index : index + num_envs] for index in range(0, len(episodes), num_envs)]
    episode_progress = tqdm(total=len(episodes), desc="Collect teacher dataset", disable=not progress)
    try:
        for batch in batches:
            episode_rows.extend(
                _collect_batch(
                    teacher,
                    batch,
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
                    episode_progress,
                )
            )
    finally:
        episode_progress.close()
    lap(collect_clock, "collect_batches")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_name": teacher_name,
        "teacher_checkpoint_path": str(teacher_path),
        "teacher_score": teacher_metadata.get("score"),
        "epsilon": epsilon,
        "seeds": [int(seed) for seed in seeds],
        "worlds": list(selected_worlds),
        "max_steps": max_steps,
        "num_envs": num_envs,
        "frames": len(observations),
        "episodes": len(episode_rows),
        "episode_rows": episode_rows,
    }
    path = cfg.dataset_path(dataset_name or _dataset_name(teacher_name, epsilon, seeds, selected_worlds))
    arrays = {
        "observations": np.asarray(observations, dtype=np.float32),
        "teacher_q_values": np.asarray(teacher_q_values, dtype=np.float32),
        "teacher_actions": np.asarray(teacher_actions, dtype=np.int64),
        "rollout_actions": np.asarray(rollout_actions, dtype=np.int64),
        "worlds": np.asarray(world_labels),
        "seeds": np.asarray(seed_values, dtype=np.int64),
        "steps": np.asarray(step_values, dtype=np.int64),
        "scenarios": np.asarray(scenario_labels),
    }
    lap(collect_clock, "prepare_arrays")
    save_dataset(path, metadata=metadata, **arrays)
    stop(collect_clock, "save_dataset")
    metadata["profile"] = _profile()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DatasetRef(path=path, metadata=metadata)


def _collect_batch(
    teacher,
    episodes: Sequence[tuple[str, int]],
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
    progress,
) -> list[dict[str, float | int | str]]:
    batch_clock = get_clock("collect_batch")
    vector_env = AsyncVectorEnv(
        [partial(_make_env, world) for world, _ in episodes],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    lap(batch_clock, "create_vector_env")
    try:
        observation, _ = vector_env.reset(seed=[seed for _, seed in episodes])
        lap(batch_clock, "reset")
        active = np.ones(len(episodes), dtype=bool)
        returns = np.zeros(len(episodes), dtype=np.float64)
        steps = np.zeros(len(episodes), dtype=np.int64)
        rows = []
        for _ in range(max_steps):
            active_indexes = np.flatnonzero(active)
            if len(active_indexes) == 0:
                break
            actions = np.zeros(len(episodes), dtype=np.int64)
            q_values = _teacher_q_values(teacher, observation[active_indexes], device)
            lap(batch_clock, "teacher_forward")
            greedy_actions = np.argmax(q_values, axis=1).astype(np.int64)
            actions[active_indexes] = greedy_actions
            exploration = epsilon > 0.0 and rng.random(len(active_indexes)) < epsilon
            if np.any(exploration):
                actions[active_indexes[exploration]] = rng.integers(
                    0, vector_env.single_action_space.n, size=int(np.sum(exploration))
                )
            lap(batch_clock, "choose_actions")

            _append_frames(
                observation,
                q_values,
                greedy_actions,
                actions,
                active_indexes,
                episodes,
                steps,
                observations,
                teacher_q_values,
                teacher_actions,
                rollout_actions,
                world_labels,
                seed_values,
                step_values,
                scenario_labels,
            )
            lap(batch_clock, "append_frames")
            observation, rewards, terminated, truncated, _ = vector_env.step(actions)
            lap(batch_clock, "env_step")
            returns[active_indexes] += rewards[active_indexes]
            steps[active_indexes] += 1
            finished = active & (terminated | truncated)
            for index in np.flatnonzero(finished):
                rows.append(_episode_row(episodes[index], steps[index], returns[index]))
                progress.update(1)
            active[finished] = False
        for index in np.flatnonzero(active):
            rows.append(_episode_row(episodes[index], steps[index], returns[index]))
            progress.update(1)
        return rows
    finally:
        vector_env.close()
        stop(batch_clock, "close_vector_env")


def _teacher_q_values(teacher, observations, device) -> np.ndarray:
    state = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        return teacher(state).detach().cpu().numpy()


def _append_frames(
    observation,
    q_values,
    greedy_actions,
    actions,
    active_indexes,
    episodes,
    steps,
    observations,
    teacher_q_values,
    teacher_actions,
    rollout_actions,
    world_labels,
    seed_values,
    step_values,
    scenario_labels,
) -> None:
    for q_index, env_index in enumerate(active_indexes):
        world, seed = episodes[env_index]
        rollout_action = int(actions[env_index])
        teacher_action = int(greedy_actions[q_index])
        observations.append(np.asarray(observation[env_index], dtype=np.float32))
        teacher_q_values.append(q_values[q_index].astype(np.float32))
        teacher_actions.append(teacher_action)
        rollout_actions.append(rollout_action)
        world_labels.append(world)
        seed_values.append(seed)
        step_values.append(int(steps[env_index]))
        scenario_labels.append("epsilon" if rollout_action != teacher_action else "greedy")


def _episode_row(episode: tuple[str, int], steps: int, score: float) -> dict[str, float | int | str]:
    world, seed = episode
    return {"world": world, "seed": int(seed), "steps": int(steps), "score": float(score)}


def _make_env(world_name: str):
    worlds = {world.name: world for world in WORLDS}
    world = worlds[world_name]
    return EnvWrapper(
        gym.make(
            "LunarLander-v3",
            gravity=world.gravity,
            enable_wind=world.wind_power[1] > 0 or world.turbulence_power[1] > 0,
        ),
        world,
        "10d",
    )


def _profile() -> dict[str, object]:
    return {
        "total_seconds": total_time("collect_teacher_dataset_parallel"),
        "lap_seconds": total_lap_times("collect_teacher_dataset_parallel"),
        "batch_total_seconds": total_time("collect_batch"),
        "batch_lap_seconds": total_lap_times("collect_batch"),
    }
