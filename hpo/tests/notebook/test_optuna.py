import optuna

from hpo.notebook.optuna import neighbors, optuna_db_summary


def test_neighbors_returns_value_plus_direct_neighbors() -> None:
    assert neighbors(10_000, [2_500, 5_000, 10_000, 20_000]) == [
        5_000,
        10_000,
        20_000,
    ]


def test_optuna_db_summary_returns_study_attrs(tmp_path) -> None:
    db_path = tmp_path / "studies.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(study_name="s1", storage=storage)
    study.set_user_attr("robust_best_score", 123.0)
    study.set_user_attr("robust_best_params", {"eps_end": 0.05})
    study.set_user_attr("incumbent_score", 120.0)
    study.set_user_attr("incumbent_params", {"eps_end": 0.04})

    summary = optuna_db_summary(db_path)

    assert summary.to_dict("records") == [{
        "study": "s1",
        "robust_best_score": 123.0,
        "robust_best_params": {"eps_end": 0.05},
        "incumbent_score": 120.0,
        "incumbent_params": {"eps_end": 0.04},
    }]
