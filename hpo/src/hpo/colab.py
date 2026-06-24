"""Convention-based Colab setup for HPO notebooks."""

from dataclasses import dataclass
from pathlib import Path

from hpo.drive_backup import backup_to_drive, restore_from_drive
from hpo.lunar_lander.logging import configure_file_logging


@dataclass(frozen=True)
class ColabSetup:
    drive_study_dir: Path
    local_study_dir: Path


@dataclass(frozen=True)
class RunStorage:
    database_path: Path
    drive_database_path: Path
    log_path: Path
    drive_log_path: Path

    def backup(self) -> None:
        backup_to_drive(
            local_database=self.database_path,
            drive_database=self.drive_database_path,
            local_log=self.log_path,
            drive_log=self.drive_log_path,
        )


def setup_colab() -> ColabSetup:
    from google.colab import drive

    drive.mount("/content/drive")
    setup = ColabSetup(
        drive_study_dir=Path("/content/drive/MyDrive/rl_lab/hpo"),
        local_study_dir=Path("/content/rl_lab/hpo/runs"),
    )
    setup.drive_study_dir.mkdir(parents=True, exist_ok=True)
    setup.local_study_dir.mkdir(parents=True, exist_ok=True)
    return setup


def prepare_run_storage(setup: ColabSetup, run_name: str) -> RunStorage:
    storage = RunStorage(
        database_path=setup.local_study_dir / f"{run_name}.db",
        drive_database_path=setup.drive_study_dir / f"{run_name}.db",
        log_path=setup.local_study_dir / f"{run_name}.log",
        drive_log_path=setup.drive_study_dir / f"{run_name}.log",
    )
    restore_from_drive(storage.drive_database_path, storage.database_path)
    restore_from_drive(storage.drive_log_path, storage.log_path)
    configure_file_logging(setup.local_study_dir, storage.log_path.name)
    return storage
