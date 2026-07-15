"""Convention-based Colab setup for HPO notebooks."""

import logging
from dataclasses import dataclass
from pathlib import Path
import shutil
import sqlite3

from hpo._logging import configure_file_logging


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


def restore_from_drive(drive_path: str | Path, local_path: str | Path) -> None:
    """Restore a missing local artifact from Drive."""
    drive_path = Path(drive_path)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if drive_path.exists() and not local_path.exists():
        shutil.copy2(drive_path, local_path)


def backup_to_drive(
    *, local_database: str | Path, drive_database: str | Path, local_log: str | Path, drive_log: str | Path
) -> None:
    """Back up the local database.

    Log without interrupting training if an error occurs.
    """
    logger = logging.getLogger(__name__)
    try:
        _backup_sqlite(local_database, drive_database)
    except (OSError, sqlite3.Error) as error:
        logger.warning("database backup failed: %s", error)

    try:
        _replace_with_copy(local_log, drive_log)
    except OSError as error:
        logger.warning("log backup failed: %s", error)


def _backup_sqlite(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    try:
        source_connection = sqlite3.connect(source)
        try:
            destination_connection = sqlite3.connect(temporary)
            try:
                source_connection.backup(destination_connection)
            finally:
                destination_connection.close()
        finally:
            source_connection.close()

        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_with_copy(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    if not source.exists():
        return

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def path(name: str | Path, google_drive: bool = False, folder: str | Path = "rl_lab/hpo") -> Path:
    """Return a local or Google Drive path."""
    root = Path("/content/drive/MyDrive") if google_drive else Path("/content")
    name = str(name)
    return root / folder / name
