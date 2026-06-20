"""Copy HPO run files between Colab and Google Drive."""

import logging
from pathlib import Path
import shutil
import sqlite3


logger = logging.getLogger(__name__)


def restore_from_drive(drive_path: str | Path, local_path: str | Path) -> None:
    """Restore a local artifact when a backup exists."""
    drive_path = Path(drive_path)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if drive_path.exists() and not local_path.exists():
        shutil.copy2(drive_path, local_path)


def backup_to_drive(
    *,
    local_database: str | Path,
    drive_database: str | Path,
    local_log: str | Path,
    drive_log: str | Path,
) -> None:
    """Back up the local database and log without interrupting training."""
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
