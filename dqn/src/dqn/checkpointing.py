"""Checkpoint helpers for saving and resuming DQN trainer state."""

from collections import deque
import copy
import logging
from pathlib import Path
import random
from typing import Any

import torch

from dqn.training import ReplayMemory, Trainer


CHECKPOINT_VERSION = 1
logger = logging.getLogger(__name__)


def save_checkpoint(trainer: Trainer, path: str | Path) -> None:
    """Save enough trainer state to resume training."""
    logger.info("Saving checkpoint to %s", path)
    torch.save(_trainer_state(trainer), path)


def load_checkpoint(trainer: Trainer, path: str | Path) -> None:
    """Load trainer state saved by save_checkpoint()."""
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)

    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {checkpoint.get('version')}")

    trainer_state = checkpoint["trainer"]
    trainer.q_net.load_state_dict(trainer_state["q_net"])
    trainer.target_net.load_state_dict(trainer_state["target_net"])
    trainer.optimizer.load_state_dict(trainer_state["optimizer"])
    trainer.steps_done = trainer_state["steps_done"]

    _load_replay_memory(trainer, checkpoint["replay_memory"])
    _load_tuned_state(trainer, checkpoint.get("tuned", {}))
    _load_rng_state(trainer, checkpoint["rng"])


def _trainer_state(trainer: Trainer) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "trainer": {
            "q_net": trainer.q_net.state_dict(),
            "target_net": trainer.target_net.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "steps_done": trainer.steps_done,
        },
        "tuned": _tuned_state(trainer),
        "replay_memory": _replay_memory_state(trainer),
        "rng": _rng_state(trainer),
    }


def _tuned_state(trainer: Trainer) -> dict[str, Any]:
    state = {}
    if hasattr(trainer, "best_checkpoint_score"):
        state["best_checkpoint_score"] = trainer.best_checkpoint_score
    if hasattr(trainer, "checkpoint_returns"):
        state["checkpoint_returns"] = list(trainer.checkpoint_returns)
    return state


def _load_tuned_state(trainer: Trainer, state: dict[str, Any]) -> None:
    if hasattr(trainer, "best_checkpoint_score") and "best_checkpoint_score" in state:
        trainer.best_checkpoint_score = state["best_checkpoint_score"]
    if hasattr(trainer, "checkpoint_returns") and "checkpoint_returns" in state:
        trainer.checkpoint_returns = list(state["checkpoint_returns"])


def _replay_memory_state(trainer: Trainer) -> dict[str, Any]:
    return {
        "capacity": trainer.memory.memory.maxlen,
        "transitions": list(trainer.memory.memory),
    }


def _load_replay_memory(trainer: Trainer, state: dict[str, Any]) -> None:
    memory = ReplayMemory(state["capacity"])
    memory.memory = deque(state["transitions"], maxlen=state["capacity"])
    trainer.memory = memory


def _rng_state(trainer: Trainer) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "env": _numpy_rng_state(trainer.env),
        "action_space": _numpy_rng_state(trainer.env.action_space),
        "observation_space": _numpy_rng_state(trainer.env.observation_space),
    }


def _load_rng_state(trainer: Trainer, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])

    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])

    _load_numpy_rng_state(trainer.env, state["env"])
    _load_numpy_rng_state(trainer.env.action_space, state["action_space"])
    _load_numpy_rng_state(trainer.env.observation_space, state["observation_space"])


def _numpy_rng_state(obj) -> dict[str, Any] | None:
    rng = getattr(obj, "np_random", None)
    if rng is None or not hasattr(rng, "bit_generator"):
        return None
    return copy.deepcopy(rng.bit_generator.state)


def _load_numpy_rng_state(obj, state: dict[str, Any] | None) -> None:
    if state is None:
        return

    rng = getattr(obj, "np_random", None)
    if rng is not None and hasattr(rng, "bit_generator"):
        rng.bit_generator.state = state
