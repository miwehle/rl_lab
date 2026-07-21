import json

import numpy as np

from distillation.dataset import DatasetRef, dataset_arrays, load_dataset, save_dataset


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
