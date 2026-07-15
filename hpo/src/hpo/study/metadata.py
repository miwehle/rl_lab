"""Store compact HPO metadata beside Optuna's own SQLite tables."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def record_study_metadata(
    database_path: str | Path, study_name: str, *, runtime_provider: str | None = None, device: Any = None
) -> None:
    """Create or update the HPO metadata row for one study."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _runtime_metadata(device=device, runtime_provider=runtime_provider)
    provider = runtime_provider or _detect_runtime_provider()
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hpo_study_metadata (
                study_name TEXT PRIMARY KEY,
                runtime_provider TEXT NOT NULL,
                runtime_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)
        connection.execute(
            """
            INSERT OR IGNORE INTO hpo_study_metadata (
                study_name,
                runtime_provider,
                runtime_metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (study_name, provider, json.dumps(metadata, sort_keys=True), datetime.now(UTC).isoformat()),
        )


def _runtime_metadata(*, device: Any = None, runtime_provider: str | None = None) -> dict[str, Any]:
    """Collect the compact runtime facts that help explain study results."""
    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "torch_version": _package_version("torch"),
        "optuna_version": _package_version("optuna"),
        "device": _device_name(device),
        "accelerator_backend": None,
        "accelerator_name": None,
        "accelerator_count": 0,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git_dirty(),
    }
    _add_accelerator_metadata(metadata)
    if (runtime_provider or _detect_runtime_provider()) == "colab":
        metadata["colab"] = {"hardware_accelerator": _colab_hardware_accelerator(metadata)}
    return metadata


def _detect_runtime_provider() -> str:
    """Return a coarse runtime provider label."""
    if "google.colab" in sys.modules:
        return "colab"
    if os.environ.get("RUNPOD_POD_ID"):
        return "runpod"
    return "local"


def _device_name(device: Any) -> str:
    if device is not None:
        return str(device)
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _add_accelerator_metadata(metadata: dict[str, Any]) -> None:
    torch = _torch()
    if torch is None:
        return
    if torch.cuda.is_available():
        metadata["accelerator_backend"] = "cuda"
        metadata["accelerator_name"] = torch.cuda.get_device_name(0)
        metadata["accelerator_count"] = torch.cuda.device_count()


def _colab_hardware_accelerator(metadata: dict[str, Any]) -> str:
    name = metadata["accelerator_name"]
    if not name:
        return "CPU"
    if "L4" in name:
        return "L4 GPU"
    if "T4" in name:
        return "T4 GPU"
    if "A100" in name:
        return "A100 GPU"
    if "H100" in name:
        return "H100 GPU"
    return name


def _cpu_name() -> str:
    return platform.processor() or platform.machine()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status)
