"""Train student networks from teacher datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dqn.model import DQN
from dqn.training import resolve_device
from hpo.checkpointing import save_checkpoint

from distillation.dataset import DatasetRef, dataset_arrays
from distillation.infra import InfraCfg


@dataclass(frozen=True)
class StudentRef:
    checkpoint_path: Path
    metadata: dict


def train_student(
    dataset: DatasetRef,
    *,
    hidden_sizes: tuple[int, int] = (64, 64),
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    run_name: str | None = None,
    seed: int = 0,
    device=None,
    cfg: InfraCfg = InfraCfg(),
) -> StudentRef:
    """Train a student to imitate stored teacher Q-values."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    cfg.prepare()
    device = resolve_device(device)
    arrays = dataset_arrays(dataset)
    observations = arrays["observations"].astype(np.float32)
    teacher_q_values = arrays["teacher_q_values"].astype(np.float32)
    train_idx, val_idx = _split_indices(len(observations), validation_fraction=validation_fraction, seed=seed)

    student = DQN(observations.shape[1], teacher_q_values.shape[1], hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    train_loader = _loader(observations[train_idx], teacher_q_values[train_idx], batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(1, epochs + 1):
        student.train()
        losses = []
        for obs_batch, q_batch in train_loader:
            obs_batch = obs_batch.to(device)
            q_batch = q_batch.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(student(obs_batch), q_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = _validation_metrics(student, observations[val_idx], teacher_q_values[val_idx], device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val})

    run_name = run_name or _run_name(dataset, hidden_sizes)
    checkpoint_path = cfg.student_checkpoint_path(run_name)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "student_hidden_sizes": list(hidden_sizes),
        "dataset_path": str(dataset.path),
        "dataset_metadata": dataset.metadata,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "history": history,
        "train_loss": history[-1]["train_loss"],
        "val_loss": history[-1]["val_loss"],
        "val_argmax_agreement": history[-1]["val_argmax_agreement"],
    }
    save_checkpoint(student, checkpoint_path, metadata)
    _write_json(checkpoint_path.parent / "training_summary.json", metadata)
    return StudentRef(checkpoint_path=checkpoint_path, metadata=metadata)


def _split_indices(count: int, *, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indexes = np.arange(count)
    rng.shuffle(indexes)
    val_count = max(1, int(round(count * validation_fraction)))
    val_count = min(val_count, count - 1)
    return indexes[val_count:], indexes[:val_count]


def _loader(obs: np.ndarray, q_values: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(obs), torch.from_numpy(q_values))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _validation_metrics(student: DQN, obs: np.ndarray, q_values: np.ndarray, device) -> dict[str, float]:
    student.eval()
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).to(device)
        teacher_tensor = torch.from_numpy(q_values).to(device)
        student_q = student(obs_tensor)
        loss = F.mse_loss(student_q, teacher_tensor)
        agreement = (student_q.argmax(dim=1) == teacher_tensor.argmax(dim=1)).float().mean()
    return {"val_loss": float(loss.cpu()), "val_argmax_agreement": float(agreement.cpu())}


def _run_name(dataset: DatasetRef, hidden_sizes: tuple[int, int]) -> str:
    teacher = dataset.metadata.get("teacher_name", "teacher")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{teacher}_student_{hidden_sizes[0]}x{hidden_sizes[1]}_{timestamp}"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
