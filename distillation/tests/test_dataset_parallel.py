import json

import numpy as np
import torch

from hpo.environments.solar_system_lander.env import World

from distillation.dataset import dataset_arrays
from distillation.dataset_parallel import collect_teacher_dataset_parallel
from distillation.infra_cfg import InfraCfg


def test_collect_teacher_dataset_parallel_writes_weighted_world_mix(monkeypatch, tmp_path):
    monkeypatch.setattr("distillation.dataset_parallel.AsyncVectorEnv", FakeVectorEnv)
    monkeypatch.setattr("distillation.dataset_parallel.checkpoint_metadata", lambda path: {"score": 264})
    monkeypatch.setattr("distillation.dataset_parallel._load_teacher", lambda *args, **kwargs: FakeTeacher())
    cfg = InfraCfg(
        teacher_archive_dir=tmp_path / "teachers",
        local_distillation_dir=tmp_path / "local",
        drive_distillation_dir=tmp_path / "drive",
    )

    dataset = collect_teacher_dataset_parallel(
        epsilon=0.0,
        seeds=[7],
        world_mix={World.VENUS: 2, World.EARTH: 1},
        num_envs=2,
        cfg=cfg,
        progress=False,
    )
    arrays = dataset_arrays(dataset)

    assert dataset.metadata["worlds"] == ["venus", "venus", "earth"]
    assert dataset.metadata["episodes"] == 3
    assert dataset.metadata["num_envs"] == 2
    assert set(dataset.metadata["profile"]) == {
        "batch_lap_seconds",
        "batch_total_seconds",
        "lap_seconds",
        "total_seconds",
    }
    assert [row["world"] for row in dataset.metadata["episode_rows"]] == ["venus", "venus", "earth"]
    saved_metadata = json.loads(dataset.path.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved_metadata["episodes"] == 3
    assert "profile" in saved_metadata
    np.testing.assert_array_equal(arrays["worlds"], np.array(["venus", "venus", "earth"]))
    assert arrays["observations"].shape == (3, 10)
    assert arrays["teacher_q_values"].shape == (3, 4)


class FakeTeacher:
    def __call__(self, observation):
        return torch.arange(observation.shape[0] * 4, dtype=torch.float32).reshape(observation.shape[0], 4)


class FakeActionSpace:
    n = 4


class FakeVectorEnv:
    single_action_space = FakeActionSpace()

    def __init__(self, env_fns, *, autoreset_mode):
        self.env_count = len(env_fns)
        self.step_count = 0

    def reset(self, *, seed):
        return np.zeros((self.env_count, 10), dtype=np.float32), {}

    def step(self, actions):
        self.step_count += 1
        observation = np.full((self.env_count, 10), self.step_count, dtype=np.float32)
        reward = np.ones(self.env_count, dtype=np.float32)
        terminated = np.ones(self.env_count, dtype=bool)
        truncated = np.zeros(self.env_count, dtype=bool)
        return observation, reward, terminated, truncated, {}

    def close(self):
        pass
