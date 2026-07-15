"""Study infrastructure conventions."""

from hpo.study_infra import StudyInfraCfg


def test_study_infra_uses_conventional_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("hpo.study_infra.configure_file_logging", lambda *_args, **_kwargs: None)
    cfg = StudyInfraCfg(drive_study_dir=tmp_path / "drive", local_study_dir=tmp_path / "local")

    (tmp_path / "drive").mkdir()
    (tmp_path / "drive" / "elise.db").write_text("db", encoding="utf-8")
    (tmp_path / "drive" / "elise.log").write_text("log", encoding="utf-8")

    storage = cfg.storage("elise")

    assert storage.database_path == tmp_path / "local" / "elise.db"
    assert storage.drive_database_path == tmp_path / "drive" / "elise.db"
    assert storage.log_path == tmp_path / "local" / "elise.log"
    assert storage.drive_log_path == tmp_path / "drive" / "elise.log"
    assert storage.database_path.read_text(encoding="utf-8") == "db"
    assert storage.log_path.read_text(encoding="utf-8") == "log"


def test_study_infra_checkpoint_conventions(tmp_path) -> None:
    cfg = StudyInfraCfg(drive_study_dir=tmp_path / "drive", local_study_dir=tmp_path / "local")

    assert cfg.checkpoint_dir("elise") == tmp_path / "local" / "elise_checkpoints"
    assert cfg.best_eval_archive_dir("elise") == tmp_path / "drive" / "best_checkpoints" / "elise"
    assert (
        cfg.best_eval_checkpoint_path("elise")
        == tmp_path / "drive" / "best_checkpoints" / "elise" / "best_eval_checkpoint.pt"
    )
