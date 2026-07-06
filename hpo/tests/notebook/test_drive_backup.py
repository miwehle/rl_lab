import sqlite3

from hpo.notebook import colab
from hpo.notebook.colab import backup_to_drive, restore_from_drive


def test_backup_to_and_restore_from_drive(tmp_path) -> None:
    local_dir = tmp_path / "local"
    backup_dir = tmp_path / "backup"
    local_dir.mkdir()
    database = local_dir / "study.db"
    log = local_dir / "study.log"

    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE values_table (value INTEGER)")
        connection.execute("INSERT INTO values_table VALUES (42)")
    log.write_text("training log\n")

    drive_database = backup_dir / "study.db"
    drive_log = backup_dir / "study.log"
    backup_to_drive(
        local_database=database, drive_database=drive_database, local_log=log, drive_log=drive_log
    )

    with sqlite3.connect(drive_database) as connection:
        assert connection.execute("SELECT value FROM values_table").fetchone() == (42,)
    assert drive_log.read_text() == "training log\n"

    restored = tmp_path / "restored" / "study.db"
    restore_from_drive(drive_database, restored)
    with sqlite3.connect(restored) as connection:
        assert connection.execute("SELECT value FROM values_table").fetchone() == (42,)


def test_database_backup_failure_does_not_stop_log_backup(tmp_path, monkeypatch) -> None:
    log = tmp_path / "study.log"
    drive_log = tmp_path / "backup" / "study.log"
    log.write_text("still running\n")

    def fail_database_backup(*_args) -> None:
        raise OSError("drive unavailable")

    monkeypatch.setattr(colab, "_backup_sqlite", fail_database_backup)

    backup_to_drive(
        local_database=tmp_path / "study.db",
        drive_database=tmp_path / "backup" / "study.db",
        local_log=log,
        drive_log=drive_log,
    )

    assert drive_log.read_text() == "still running\n"
