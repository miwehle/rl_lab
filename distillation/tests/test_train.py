import numpy as np

from distillation.dataset import save_dataset
from distillation.infra_cfg import InfraCfg
from distillation.train import train_student


def test_train_student_smoke_test_writes_checkpoint_and_metadata(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    observations = np.random.default_rng(0).normal(size=(32, 10)).astype(np.float32)
    teacher_q_values = observations[:, :4] * 0.5
    dataset = save_dataset(
        dataset_path,
        metadata={"teacher_name": "teacher", "frames": 32},
        observations=observations,
        teacher_q_values=teacher_q_values,
        teacher_actions=teacher_q_values.argmax(axis=1),
        rollout_actions=teacher_q_values.argmax(axis=1),
        worlds=np.array(["moon"] * 32),
        seeds=np.arange(32, dtype=np.int64),
        steps=np.zeros(32, dtype=np.int64),
        scenarios=np.array(["synthetic"] * 32),
    )
    cfg = InfraCfg(
        teacher_archive_dir=tmp_path / "teachers",
        local_distillation_dir=tmp_path / "local",
        drive_distillation_dir=tmp_path / "drive",
    )

    student = train_student(dataset, hidden_sizes=(8, 6), epochs=2, batch_size=8, run_name="run", cfg=cfg)

    assert student.checkpoint_path == tmp_path / "drive" / "runs" / "run" / "student_checkpoint.pt"
    assert student.checkpoint_path.exists()
    assert student.checkpoint_path.with_suffix(".json").exists()
    assert (student.checkpoint_path.parent / "training_summary.json").exists()
    assert student.metadata["student_hidden_sizes"] == [8, 6]
    assert "val_argmax_agreement" in student.metadata
