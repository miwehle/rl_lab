"""Q-network checkpoint helpers without an HPO package dependency."""

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dqn.model import DQN
from dqn.training import ModelFactory, resolve_device
from dqn.vector_training import VectorTrainingConfig


CHECKPOINT_VERSION = 1


def save_q_net_checkpoint(q_net: nn.Module, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"version": CHECKPOINT_VERSION, "model_state_dict": q_net.state_dict(), "metadata": metadata or {}},
        path,
    )


def load_q_net_checkpoint(q_net: nn.Module, path: str | Path, device=None) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {checkpoint.get('version')}")

    q_net.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint["metadata"]


def q_net_from_checkpoint(path: str | Path, *, make_env, device=None, model_factory: ModelFactory = DQN):
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
        load_q_net_checkpoint(q_net, path, device)
        return q_net
    finally:
        env.close()


def vector_training_config_from_checkpoint(path: str | Path, **overrides) -> VectorTrainingConfig:
    training_config = _checkpoint_training_config(path)
    if not training_config:
        raise ValueError("checkpoint metadata has no training_config")
    return VectorTrainingConfig(**(training_config | overrides))


def _checkpoint_hidden_size(path: str | Path) -> int:
    training_config = _checkpoint_training_config(path)
    return int(training_config.get("hidden_size", 128))


def _checkpoint_training_config(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    return metadata.get("training_config", {})
