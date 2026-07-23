"""Runtime input ablations for Elise-like DQN networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from dqn.model import DQN


@dataclass(frozen=True)
class InputAblation:
    name: str
    zero_indexes: tuple[int, ...]


DEFAULT_INPUT_ABLATIONS = (
    InputAblation("normal", ()),
    InputAblation("ax=0", (8,)),
    InputAblation("ay=0", (9,)),
    InputAblation("ax+ay=0", (8, 9)),
)


def evaluate_input_ablations(
    q_net: DQN,
    env_factory: Any,
    worlds: Iterable[str],
    seeds: Iterable[int],
    *,
    ablations: Iterable[InputAblation] = DEFAULT_INPUT_ABLATIONS,
    max_steps: int = 1000,
    device: Any = "cpu",
) -> tuple[dict[str, float | int | str], ...]:
    """Evaluate greedy rollouts while zeroing selected input components."""
    q_net.eval()
    worlds = tuple(worlds)
    seeds = tuple(seeds)
    ablations = tuple(ablations)
    episode_rows = [
        _evaluate_episode(q_net, env_factory, world, seed, ablation, max_steps=max_steps, device=device)
        for world in worlds
        for seed in seeds
        for ablation in ablations
    ]
    return _summarize(episode_rows)


def _evaluate_episode(
    q_net: DQN,
    env_factory: Any,
    world: str,
    seed: int,
    ablation: InputAblation,
    *,
    max_steps: int,
    device: Any,
) -> dict[str, float | int | str]:
    env = env_factory.make_env(world)
    observation, _ = env.reset(seed=seed)
    score = 0.0
    agreements = 0
    q_delta_sum = 0.0
    steps = 0
    try:
        for step in range(max_steps):
            normal_q = _q_values(q_net, observation, device)
            ablated_observation = _ablate(observation, ablation.zero_indexes)
            ablated_q = _q_values(q_net, ablated_observation, device)
            normal_action = int(np.argmax(normal_q))
            action = int(np.argmax(ablated_q))
            agreements += int(action == normal_action)
            q_delta_sum += float(np.mean(np.abs(ablated_q - normal_q)))
            observation, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            steps = step + 1
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "world": world,
        "seed": int(seed),
        "ablation": ablation.name,
        "score": score,
        "steps": steps,
        "action_agreement": agreements / steps if steps else 0.0,
        "q_delta": q_delta_sum / steps if steps else 0.0,
    }


def _ablate(observation: np.ndarray, zero_indexes: tuple[int, ...]) -> np.ndarray:
    ablated = np.asarray(observation, dtype=np.float32).copy()
    ablated[list(zero_indexes)] = 0.0
    return ablated


def _q_values(q_net: DQN, observation: np.ndarray, device: Any) -> np.ndarray:
    x = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return q_net(x)[0].cpu().numpy()


def _summarize(episode_rows: list[dict[str, float | int | str]]) -> tuple[dict[str, float | int | str], ...]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in episode_rows:
        groups.setdefault((str(row["world"]), str(row["ablation"])), []).append(row)

    normal_scores = {
        world: float(np.mean([float(row["score"]) for row in rows]))
        for (world, ablation), rows in groups.items()
        if ablation == "normal"
    }
    summary = []
    for (world, ablation), rows in groups.items():
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "world": world,
                "ablation": ablation,
                "episodes": len(rows),
                "mean_score": float(np.mean(scores)),
                "delta_vs_normal": float(np.mean(scores) - normal_scores[world]),
                "mean_steps": float(np.mean([int(row["steps"]) for row in rows])),
                "action_agreement": float(np.mean([float(row["action_agreement"]) for row in rows])),
                "q_delta": float(np.mean([float(row["q_delta"]) for row in rows])),
            }
        )
    return tuple(summary)
