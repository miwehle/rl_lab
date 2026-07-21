from pathlib import Path

from distillation.infra_cfg import InfraCfg


def test_infra_cfg_derives_conventional_paths(tmp_path):
    cfg = InfraCfg(
        teacher_archive_dir=tmp_path / "teachers",
        local_distillation_dir=tmp_path / "local",
        drive_distillation_dir=tmp_path / "drive",
    )

    assert cfg.teacher_checkpoint_dir("teacher") == tmp_path / "teachers" / "teacher"
    assert cfg.teacher_checkpoint_path("teacher") == tmp_path / "teachers" / "teacher" / "best_eval_checkpoint.pt"
    assert cfg.dataset_dir() == tmp_path / "drive" / "datasets"
    assert cfg.dataset_path("dataset") == tmp_path / "drive" / "datasets" / "dataset.npz"
    assert cfg.runs_dir() == tmp_path / "drive" / "runs"
    assert cfg.run_dir("run") == tmp_path / "drive" / "runs" / "run"
    assert cfg.student_checkpoint_path("run") == tmp_path / "drive" / "runs" / "run" / "student_checkpoint.pt"


def test_infra_cfg_prepare_creates_artifact_dirs(tmp_path):
    cfg = InfraCfg(
        teacher_archive_dir=Path("/unused"),
        local_distillation_dir=tmp_path / "local",
        drive_distillation_dir=tmp_path / "drive",
    )

    cfg.prepare()

    assert cfg.local_distillation_dir.is_dir()
    assert cfg.dataset_dir().is_dir()
    assert cfg.runs_dir().is_dir()
