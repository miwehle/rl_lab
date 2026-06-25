"""Convention-based Colab setup for HPO notebooks."""

from dataclasses import dataclass
from pathlib import Path

from hpo.drive_backup import backup_to_drive, restore_from_drive
from hpo.lunar_lander.logging import configure_file_logging


@dataclass(frozen=True)
class ColabSetup:
    """Shared local and Drive directories for Colab HPO notebooks."""

    drive_study_dir: Path
    local_study_dir: Path


@dataclass(frozen=True)
class Storage:
    """Paths for one SQLite database and its log file.

    Each file has a local path and a Drive backup path.
    """

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
    """Mount Drive and create the conventional HPO work directories."""

    from google.colab import drive

    drive.mount("/content/drive")
    setup = ColabSetup(
        drive_study_dir=Path("/content/drive/MyDrive/rl_lab/hpo"),
        local_study_dir=Path("/content/rl_lab/hpo/runs"),
    )
    setup.drive_study_dir.mkdir(parents=True, exist_ok=True)
    setup.local_study_dir.mkdir(parents=True, exist_ok=True)
    return setup


def prepare_storage(setup: ColabSetup, storage_name: str) -> Storage:
    """Prepare database and log files using storage_name as file basename."""

    storage = Storage(
        database_path=setup.local_study_dir / f"{storage_name}.db",
        drive_database_path=setup.drive_study_dir / f"{storage_name}.db",
        log_path=setup.local_study_dir / f"{storage_name}.log",
        drive_log_path=setup.drive_study_dir / f"{storage_name}.log",
    )
    restore_from_drive(storage.drive_database_path, storage.database_path)
    restore_from_drive(storage.drive_log_path, storage.log_path)
    configure_file_logging(setup.local_study_dir, storage.log_path.name)
    return storage
