import json

import numpy as np

from hpo.environments.solar_system_lander.env import World

from distillation.dataset import DatasetRef, collect_teacher_dataset, dataset_arrays, load_dataset, save_dataset
from distillation.infra_cfg import InfraCfg


def test_dataset_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "dataset.npz"
    metadata = {"teacher_name": "teacher", "frames": 3}
    observations = np.arange(30, dtype=np.float32).reshape(3, 10)
    teacher_q_values = np.arange(12, dtype=np.float32).reshape(3, 4)

    saved = save_dataset(
        path,
        metadata=metadata,
        observations=observations,
        teacher_q_values=teacher_q_values,
        teacher_actions=np.array([0, 1, 2], dtype=np.int64),
        rollout_actions=np.array([0, 1, 3], dtype=np.int64),
        worlds=np.array(["moon", "earth", "venus"]),
        seeds=np.array([7, 42, 1911], dtype=np.int64),
        steps=np.array([0, 1, 2], dtype=np.int64),
        scenarios=np.array(["greedy", "epsilon", "greedy"]),
    )

    loaded = load_dataset(path)
    arrays = dataset_arrays(loaded)

    assert saved == DatasetRef(path=path, metadata=metadata)
    assert loaded == saved
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) == metadata
    np.testing.assert_array_equal(arrays["observations"], observations)
    np.testing.assert_array_equal(arrays["teacher_q_values"], teacher_q_values)


def test_collect_teacher_dataset_accepts_weighted_world_mix(monkeypatch, tmp_path):
    monkeypatch.setattr("distillation.dataset.EnvFactory", FakeEnvFactory)
    monkeypatch.setattr("distillation.dataset.checkpoint_metadata", lambda path: {"score": 264})
    monkeypatch.setattr("distillation.dataset._load_teacher", lambda *args, **kwargs: object())
    monkeypatch.setattr("distillation.dataset._collect_episode", fake_collect_episode)
    cfg = InfraCfg(
        teacher_archive_dir=tmp_path / "teachers",
        local_distillation_dir=tmp_path / "local",
        drive_distillation_dir=tmp_path / "drive",
    )

    dataset = collect_teacher_dataset(
        epsilon=0.0,
        seeds=[7],
        world_mix={World.VENUS: 2, World.EARTH: 1},
        cfg=cfg,
        progress=False,
    )

    assert dataset.metadata["worlds"] == ["venus", "venus", "earth"]
    assert dataset.metadata["episode_rows"] == [
        {"world": "venus", "seed": 7, "steps": 1, "score": 1.0},
        {"world": "venus", "seed": 7, "steps": 1, "score": 1.0},
        {"world": "earth", "seed": 7, "steps": 1, "score": 1.0},
    ]


class FakeEnvFactory:
    def __init__(self, observation_mode, *, world_mix):
        self.observation_mode = observation_mode
        self.world_mix = world_mix

    def make_env(self, world):
        return object()


def fake_collect_episode(
    teacher,
    env,
    seed,
    world,
    epsilon,
    max_steps,
    rng,
    device,
    observations,
    teacher_q_values,
    teacher_actions,
    rollout_actions,
    world_labels,
    seed_values,
    step_values,
    scenario_labels,
):
    observations.append(np.zeros(10, dtype=np.float32))
    teacher_q_values.append(np.zeros(4, dtype=np.float32))
    teacher_actions.append(0)
    rollout_actions.append(0)
    world_labels.append(world)
    seed_values.append(seed)
    step_values.append(0)
    scenario_labels.append("synthetic")
    return {"steps": 1, "score": 1.0}
