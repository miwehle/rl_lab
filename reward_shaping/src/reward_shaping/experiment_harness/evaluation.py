"""Unshaped greedy evaluation for reward shaping experiments."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from dqn.training import resolve_device
from reward_shaping.ground_side_thrust import is_ground_side_thrust


@dataclass(frozen=True)
class EvaluationRow:
    measurement: str
    world: str
    episode: int
    seed: int | None
    score: float
    ground_side_thrust_steps: int


@dataclass(frozen=True)
class EvaluationResult:
    measurement: str
    score: float
    world_scores: dict[str, float]
    episodes_per_world: int
    eval_seed: int | None
    rows: list[EvaluationRow]


def historical_score(
    *,
    q_net: Any,
    make_envs: Mapping[str, Callable[[], Any]],
    device=None,
    eval_seed: int = 10_000,
    max_steps: int = 2_000,
) -> EvaluationResult:
    return evaluate_unshaped(
        measurement="historical_score",
        q_net=q_net,
        make_envs=make_envs,
        episodes_per_world=10,
        eval_seed=eval_seed,
        max_steps=max_steps,
        device=device,
    )


def robust_score(
    *,
    q_net: Any,
    make_envs: Mapping[str, Callable[[], Any]],
    device=None,
    episodes_per_world: int = 50,
    eval_seed: int = 10_000,
    max_steps: int = 2_000,
) -> EvaluationResult:
    return evaluate_unshaped(
        measurement="robust_score",
        q_net=q_net,
        make_envs=make_envs,
        episodes_per_world=episodes_per_world,
        eval_seed=eval_seed,
        max_steps=max_steps,
        device=device,
    )


def evaluate_unshaped(
    *,
    measurement: str,
    q_net: Any,
    make_envs: Mapping[str, Callable[[], Any]],
    episodes_per_world: int,
    eval_seed: int | None = 10_000,
    max_steps: int = 2_000,
    device=None,
) -> EvaluationResult:
    """Measure greedy mean Gym score and ground side-thrust steps."""
    if episodes_per_world < 1:
        raise ValueError("episodes_per_world must be >= 1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    device = resolve_device(device)
    q_net.eval()
    rows: list[EvaluationRow] = []

    for world, make_env in make_envs.items():
        for episode in range(episodes_per_world):
            seed = None if eval_seed is None else eval_seed + episode
            score, ground_side_thrust_steps = _episode_return(
                q_net, make_env, device, max_steps=max_steps, seed=seed
            )
            rows.append(
                EvaluationRow(
                    measurement=measurement,
                    world=world,
                    episode=episode,
                    seed=seed,
                    score=score,
                    ground_side_thrust_steps=ground_side_thrust_steps,
                )
            )

    world_scores = {
        world: _mean([row.score for row in rows if row.world == world])
        for world in make_envs
    }
    return EvaluationResult(
        measurement=measurement,
        score=_mean(list(world_scores.values())),
        world_scores=world_scores,
        episodes_per_world=episodes_per_world,
        eval_seed=eval_seed,
        rows=rows,
    )


def _episode_return(q_net, make_env, device, *, max_steps: int, seed: int | None) -> tuple[float, int]:
    env = make_env()
    try:
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        ground_side_thrust_steps = 0

        for _ in range(max_steps):
            state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).flatten(start_dim=1)
            with torch.no_grad():
                action = int(q_net(state).argmax(dim=1).item())

            if is_ground_side_thrust(env.unwrapped, action):
                ground_side_thrust_steps += 1

            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break

        return episode_return, ground_side_thrust_steps
    finally:
        env.close()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
