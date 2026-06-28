from pathlib import Path

import optuna

from hpo.notebook.optuna import db_path, db_summary, neighbors


def test_db_path_returns_colab_paths() -> None:
    assert db_path("study", True) == (
        Path("/content/drive/MyDrive/rl_lab/hpo/study.db")
    )
    assert db_path("study", google_drive=True) == (
        Path("/content/drive/MyDrive/rl_lab/hpo/study.db")
    )
    assert db_path("study.db") == Path("/content/rl_lab/hpo/study.db")
    assert db_path("study", folder="rl_lab/hpo/runs") == (
        Path("/content/rl_lab/hpo/runs/study.db")
    )


def test_neighbors_returns_value_plus_direct_neighbors() -> None:
    assert neighbors(10_000, [2_500, 5_000, 10_000, 20_000]) == [
        5_000,
        10_000,
        20_000,
    ]


def test_db_summary_returns_study_attrs(tmp_path) -> None:
    db_path = tmp_path / "studies.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(study_name="s1", storage=storage)
    study.set_user_attr("robust_best_score", 123.0)
    study.set_user_attr("robust_best_params", {"eps_end": 0.05})
    study.set_user_attr("incumbent_score", 120.0)
    study.set_user_attr("incumbent_params", {"eps_end": 0.04})

    summary = db_summary(db_path)

    assert summary.to_dict("records") == [{
        "study": "s1",
        "robust_best_score": 123.0,
        "robust_best_params": {"eps_end": 0.05},
        "incumbent_score": 120.0,
        "incumbent_params": {"eps_end": 0.04},
    }]
