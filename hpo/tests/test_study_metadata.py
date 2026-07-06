import json
import sqlite3

from hpo import study_metadata


def test_record_study_metadata_creates_hpo_table(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        study_metadata,
        "runtime_metadata",
        lambda **_kwargs: {
            "python_version": "3.12.0",
            "platform": "test",
            "cpu": "test-cpu",
            "torch_version": "2.0",
            "optuna_version": "4.0",
            "device": "cuda",
            "accelerator_backend": "cuda",
            "accelerator_name": "NVIDIA L4",
            "accelerator_count": 1,
            "git_commit": "abc",
            "git_dirty": False,
            "colab": {"hardware_accelerator": "L4 GPU"},
        },
    )
    database_path = tmp_path / "study.db"

    study_metadata.record_study_metadata(
        database_path,
        "s1",
        runtime_provider="colab",
        device="cuda",
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT study_name, runtime_provider, runtime_metadata_json, created_at
            FROM hpo_study_metadata
            """
        ).fetchone()

    assert row[0] == "s1"
    assert row[1] == "colab"
    assert json.loads(row[2])["accelerator_name"] == "NVIDIA L4"
    assert row[3]


def test_record_study_metadata_keeps_existing_row(tmp_path, monkeypatch) -> None:
    values = [
        {"python_version": "first"},
        {"python_version": "second"},
    ]
    monkeypatch.setattr(
        study_metadata,
        "runtime_metadata",
        lambda **_kwargs: values.pop(0),
    )
    database_path = tmp_path / "study.db"

    study_metadata.record_study_metadata(database_path, "s1", runtime_provider="local")
    study_metadata.record_study_metadata(database_path, "s1", runtime_provider="colab")

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT runtime_provider, runtime_metadata_json FROM hpo_study_metadata"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == "local"
    assert json.loads(rows[0][1]) == {"python_version": "first"}
