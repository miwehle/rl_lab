import numpy as np
import torch

from dqn.model import DQN
from hpo.checkpointing import save_checkpoint

from distillation.evaluate import evaluate_student
from distillation.infra_cfg import InfraCfg
from distillation.train import StudentRef


class FakeActionSpace:
    n = 4


class FakeEnv:
    action_space = FakeActionSpace()

    def __init__(self):
        self.steps = 0

    def reset(self, *, seed=None):
        self.steps = 0
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(10, dtype=np.float32), 1.5, True, False, {}

    def close(self):
        pass


class FakeEnvFactory:
    def __init__(self, observation_mode, *, world_mix):
        self.observation_mode = observation_mode
        self.world_mix = world_mix

    def make_env(self, world):
        return FakeEnv()


def test_evaluate_student_writes_summary(monkeypatch, tmp_path):
    monkeypatch.setattr("distillation.evaluate.EnvFactory", FakeEnvFactory)
    checkpoint_path = tmp_path / "drive" / "runs" / "run" / "student_checkpoint.pt"
    model = DQN(10, 4, hidden_sizes=(8, 6))
    save_checkpoint(model, checkpoint_path, {"student_hidden_sizes": [8, 6]})
    student = StudentRef(
        checkpoint_path=checkpoint_path,
        metadata={
            "student_hidden_sizes": [8, 6],
            "dataset_path": "dataset.npz",
            "dataset_metadata": {"teacher_name": "teacher"},
        },
    )
    cfg = InfraCfg(
        teacher_archive_dir=tmp_path / "teachers",
        local_distillation_dir=tmp_path / "local",
        drive_distillation_dir=tmp_path / "drive",
    )

    summary = evaluate_student(student, eval_episodes_per_world=2, worlds=("moon",), cfg=cfg)

    assert summary["episodes"] == 2
    assert summary["mean"] == 1.5
    assert summary["world_scores"] == {"moon": 1.5}
    assert summary["teacher_name"] == "teacher"
    assert (checkpoint_path.parent / "evaluation_summary.json").exists()
