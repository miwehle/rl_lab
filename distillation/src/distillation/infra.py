"""Infrastructure conventions for distillation runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_GOOGLE_DRIVE_MOUNT_CHECKED = False


@dataclass(frozen=True)
class InfraCfg:
    """Path conventions for teacher archives and distillation artifacts."""

    teacher_archive_dir: Path = Path("/content/drive/MyDrive/rl_lab/hpo/best_checkpoints")
    local_distillation_dir: Path = Path("/content/rl_lab/distillation/runs")
    drive_distillation_dir: Path = Path("/content/drive/MyDrive/rl_lab/distillation")

    def prepare(self) -> None:
        _mount_google_drive_if_available()
        self.local_distillation_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir().mkdir(parents=True, exist_ok=True)
        self.runs_dir().mkdir(parents=True, exist_ok=True)

    def teacher_checkpoint_dir(self, teacher_name: str) -> Path:
        return self.teacher_archive_dir / teacher_name

    def teacher_checkpoint_path(self, teacher_name: str) -> Path:
        return self.teacher_checkpoint_dir(teacher_name) / "best_eval_checkpoint.pt"

    def dataset_dir(self) -> Path:
        return self.drive_distillation_dir / "datasets"

    def runs_dir(self) -> Path:
        return self.drive_distillation_dir / "runs"

    def dataset_path(self, dataset_name: str) -> Path:
        return self.dataset_dir() / f"{dataset_name}.npz"

    def run_dir(self, run_name: str) -> Path:
        return self.runs_dir() / run_name

    def student_checkpoint_path(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "student_checkpoint.pt"


def _mount_google_drive_if_available() -> None:
    global _GOOGLE_DRIVE_MOUNT_CHECKED
    if _GOOGLE_DRIVE_MOUNT_CHECKED:
        return

    try:
        from google.colab import drive
    except ModuleNotFoundError:
        _GOOGLE_DRIVE_MOUNT_CHECKED = True
        return

    drive.mount("/content/drive")
    _GOOGLE_DRIVE_MOUNT_CHECKED = True
