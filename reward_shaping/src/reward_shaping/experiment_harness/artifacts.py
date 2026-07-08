"""File artifacts for reward shaping runs."""

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
import csv
import shutil
from typing import Any

import yaml

from reward_shaping.experiment_harness.evaluation import EvaluationResult


@dataclass(frozen=True)
class RunPaths:
    root: Path
    inputs: Path
    outputs: Path
    initial_checkpoint: Path
    config: Path
    training_summary: Path
    eval_scores: Path
    shaped_checkpoint: Path


def prepare_run_directory(root: str | Path, run_id: str, *, initial_checkpoint: str | Path) -> RunPaths:
    paths = run_paths(root, run_id)
    paths.inputs.mkdir(parents=True, exist_ok=True)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(initial_checkpoint, paths.initial_checkpoint)
    return paths


def run_paths(root: str | Path, run_id: str) -> RunPaths:
    run_root = Path(root) / run_id
    inputs = run_root / "inputs"
    outputs = run_root / "outputs"
    return RunPaths(
        root=run_root,
        inputs=inputs,
        outputs=outputs,
        initial_checkpoint=inputs / "initial_checkpoint.pt",
        config=outputs / "config.yaml",
        training_summary=outputs / "training_summary.yaml",
        eval_scores=outputs / "eval_scores.csv",
        shaped_checkpoint=outputs / "shaped_checkpoint.pt",
    )


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(plain_data(data), sort_keys=False), encoding="utf-8")


def write_eval_scores(path: str | Path, results: list[EvaluationResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["measurement", "world", "episode", "seed", "score", "ground_side_thrust_steps"],
        )
        writer.writeheader()
        for result in results:
            for row in result.rows:
                writer.writerow(asdict(row))


def plain_data(value: Any) -> Any:
    """Return YAML-friendly data made of plain Python containers and scalars."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain_data(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(plain_data(key)): plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    return value
