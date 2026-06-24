from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

from hpo import colab
from hpo.colab import ColabSetup, prepare_run_storage, setup_colab


def test_setup_colab_uses_conventional_directories(monkeypatch) -> None:
    mounted = []
    created = []
    drive = SimpleNamespace(mount=mounted.append)
    google_module = ModuleType("google")
    colab_module = ModuleType("google.colab")
    colab_module.drive = drive
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda self, parents=False, exist_ok=False: created.append(
            (self, parents, exist_ok)
        ),
    )

    setup = setup_colab()

    assert mounted == ["/content/drive"]
    assert setup == ColabSetup(
        drive_study_dir=Path("/content/drive/MyDrive/rl_lab/hpo"),
        local_study_dir=Path("/content/rl_lab/hpo/runs"),
    )
    assert created == [
        (setup.drive_study_dir, True, True),
        (setup.local_study_dir, True, True),
    ]


def test_prepare_run_storage_uses_conventional_files(tmp_path, monkeypatch) -> None:
    restored = []
    logging_calls = []
    setup = ColabSetup(
        drive_study_dir=tmp_path / "drive",
        local_study_dir=tmp_path / "local",
    )
    monkeypatch.setattr(colab, "restore_from_drive", lambda *args: restored.append(args))
    monkeypatch.setattr(
        colab,
        "configure_file_logging",
        lambda *args: logging_calls.append(args),
    )

    storage = prepare_run_storage(setup, "study_name")

    assert storage.database_path == setup.local_study_dir / "study_name.db"
    assert storage.drive_database_path == setup.drive_study_dir / "study_name.db"
    assert storage.log_path == setup.local_study_dir / "study_name.log"
    assert storage.drive_log_path == setup.drive_study_dir / "study_name.log"
    assert restored == [
        (storage.drive_database_path, storage.database_path),
        (storage.drive_log_path, storage.log_path),
    ]
    assert logging_calls == [(setup.local_study_dir, "study_name.log")]


def test_run_storage_backs_up_database_and_log(tmp_path, monkeypatch) -> None:
    backup_calls = []
    setup = ColabSetup(
        drive_study_dir=tmp_path / "drive",
        local_study_dir=tmp_path / "local",
    )
    monkeypatch.setattr(colab, "restore_from_drive", lambda *_args: None)
    monkeypatch.setattr(colab, "configure_file_logging", lambda *_args: None)
    monkeypatch.setattr(
        colab,
        "backup_to_drive",
        lambda **kwargs: backup_calls.append(kwargs),
    )

    prepare_run_storage(setup, "study_name").backup()

    assert backup_calls == [{
        "local_database": setup.local_study_dir / "study_name.db",
        "drive_database": setup.drive_study_dir / "study_name.db",
        "local_log": setup.local_study_dir / "study_name.log",
        "drive_log": setup.drive_study_dir / "study_name.log",
    }]
