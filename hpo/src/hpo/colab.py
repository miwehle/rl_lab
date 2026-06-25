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
class StudyStorage:
    """Database and log paths for one named HPO storage.

    The name can identify one Optuna study or a study series whose studies
    share one SQLite database.
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


class StudySeriesStorage:
    """Lazy storage helper for notebooks that write one database per study.

    A study's files are restored and logging is configured only when that
    study name is requested for the first time.
    """

    def __init__(self, setup: ColabSetup) -> None:
        self.setup = setup
        self._study_storages: dict[str, StudyStorage] = {}

    def database_path(self, study_name: str) -> Path:
        return self.study_storage(study_name).database_path

    def backup(self) -> None:
        for storage in self._study_storages.values():
            storage.backup()

    def study_storage(self, study_name: str) -> StudyStorage:
        if study_name not in self._study_storages:
            self._study_storages[study_name] = prepare_study_storage(
                self.setup,
                study_name,
            )
        return self._study_storages[study_name]


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


def prepare_study_storage(setup: ColabSetup, storage_name: str) -> StudyStorage:
    """Restore files and configure logging for one named HPO storage."""

    storage = StudyStorage(
        database_path=setup.local_study_dir / f"{storage_name}.db",
        drive_database_path=setup.drive_study_dir / f"{storage_name}.db",
        log_path=setup.local_study_dir / f"{storage_name}.log",
        drive_log_path=setup.drive_study_dir / f"{storage_name}.log",
    )
    restore_from_drive(storage.drive_database_path, storage.database_path)
    restore_from_drive(storage.drive_log_path, storage.log_path)
    configure_file_logging(setup.local_study_dir, storage.log_path.name)
    return storage


def prepare_study_series_storage(setup: ColabSetup) -> StudySeriesStorage:
    """Create lazy per-study storage for a notebook study series.

    Individual study databases are prepared only when the notebook first asks
    for that study's path.
    """

    return StudySeriesStorage(setup)
