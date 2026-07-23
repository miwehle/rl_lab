"""Collect live activations from Elise-like DQN rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from dqn.model import DQN
from hpo.checkpointing import checkpoint_metadata, load_checkpoint

ACTION_ORDER = (1, 2, 0, 3)
ACTION_LABELS = {0: "noop", 1: "left", 2: "up", 3: "right"}


@dataclass(frozen=True)
class RolloutSpec:
    world: str
    seed: int
    max_steps: int = 1000


@dataclass(frozen=True)
class ActivationRollouts:
    observations: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    q_values: np.ndarray
    actions: np.ndarray
    rows: tuple[dict[str, float | int | str], ...]

    @property
    def frame_count(self) -> int:
        return int(self.actions.shape[0])


def load_student_network(checkpoint_path: str | Path, *, device: Any = "cpu") -> DQN:
    """Load a distillation student DQN from a checkpoint and its metadata."""
    metadata = checkpoint_metadata(checkpoint_path)
    hidden_sizes = tuple(metadata["student_hidden_sizes"])
    q_net = DQN(n_observations=10, n_actions=4, hidden_sizes=hidden_sizes).to(device)
    load_checkpoint(q_net, checkpoint_path, device)
    q_net.eval()
    return q_net


def collect_activations(
    q_net: DQN, env_factory: Any, rollouts: Iterable[RolloutSpec], *, device: Any = "cpu"
) -> ActivationRollouts:
    """Run greedy episodes and collect hidden activations and Q-values for each step."""
    h1_values: list[np.ndarray] = []
    h2_values: list[np.ndarray] = []
    q_values: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rows: list[dict[str, float | int | str]] = []
    q_net.eval()

    for spec in rollouts:
        _collect_episode(
            q_net,
            env_factory.make_env(spec.world),
            spec,
            device,
            h1_values,
            h2_values,
            q_values,
            observations,
            actions,
            rows,
        )

    if not actions:
        raise ValueError("collect_activations needs at least one environment step")
    return ActivationRollouts(
        observations=np.vstack(observations),
        h1=np.vstack(h1_values),
        h2=np.vstack(h2_values),
        q_values=np.vstack(q_values),
        actions=np.asarray(actions, dtype=np.int64),
        rows=tuple(rows),
    )


def _collect_episode(
    q_net: DQN,
    env: Any,
    spec: RolloutSpec,
    device: Any,
    h1_values: list[np.ndarray],
    h2_values: list[np.ndarray],
    q_values: list[np.ndarray],
    observations: list[np.ndarray],
    actions: list[int],
    rows: list[dict[str, float | int | str]],
) -> None:
    observation, _ = env.reset(seed=spec.seed)
    score = 0.0
    try:
        for step in range(spec.max_steps):
            h1, h2, q = _forward_activations(q_net, observation, device)
            action = int(np.argmax(q))
            observations.append(np.asarray(observation, dtype=np.float32))
            h1_values.append(h1)
            h2_values.append(h2)
            q_values.append(q)
            actions.append(action)
            rows.append(
                {
                    "world": spec.world,
                    "seed": spec.seed,
                    "step": step,
                    "score_before_step": score,
                    "action": action,
                }
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()


def _forward_activations(
    q_net: DQN, observation: np.ndarray, device: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        h1 = F.relu(q_net.layer1(x))
        h2 = F.relu(q_net.layer2(h1))
        q = q_net.layer3(h2)
    return h1[0].cpu().numpy(), h2[0].cpu().numpy(), q[0].cpu().numpy()
