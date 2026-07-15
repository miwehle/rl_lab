"""Infrastructure conventions for HPO studies."""

from dataclasses import dataclass
from pathlib import Path

from hpo.notebook.colab import Storage, restore_from_drive
from hpo.lunar_lander.logging import configure_file_logging

_GOOGLE_DRIVE_MOUNT_CHECKED = False


@dataclass(frozen=True)
class InfraCfg:
    """Infrastructure conventions for one HPO study database."""

    drive_study_dir: Path = Path("/content/drive/MyDrive/rl_lab/hpo")
    local_study_dir: Path = Path("/content/rl_lab/hpo/runs")
    best_checkpoints_dir: str = "best_checkpoints"
    checkpoint_name: str = "best_eval_checkpoint.pt"

    def prepare(self) -> None:
        self._mount_google_drive_if_available()
        self.drive_study_dir.mkdir(parents=True, exist_ok=True)
        self.local_study_dir.mkdir(parents=True, exist_ok=True)

    def storage(self, storage_name: str) -> Storage:
        self.prepare()
        storage = Storage(
            database_path=self.local_study_dir / f"{storage_name}.db",
            drive_database_path=self.drive_study_dir / f"{storage_name}.db",
            log_path=self.local_study_dir / f"{storage_name}.log",
            drive_log_path=self.drive_study_dir / f"{storage_name}.log",
        )
        restore_from_drive(storage.drive_database_path, storage.database_path)
        restore_from_drive(storage.drive_log_path, storage.log_path)
        configure_file_logging(self.local_study_dir, storage.log_path.name)
        return storage

    def checkpoint_dir(self, study_name: str) -> Path:
        return self.local_study_dir / f"{study_name}_checkpoints"

    def best_eval_archive_dir(self, study_name: str) -> Path:
        return self.drive_study_dir / self.best_checkpoints_dir / study_name

    def best_eval_checkpoint_path(self, study_name: str) -> Path:
        return self.best_eval_archive_dir(study_name) / self.checkpoint_name

    def _mount_google_drive_if_available(self) -> None:
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
