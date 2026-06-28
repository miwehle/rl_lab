"""Small Optuna helpers for HPO notebooks."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def neighbors(value: Any, choices: Sequence[Any]) -> list[Any]:
    """Return value plus its direct neighbors in choices."""
    index = choices.index(value)
    return list(choices[max(0, index - 1):index + 2])


def optuna_db_summary(db_path: str | Path):
    """Return one row per study in an Optuna SQLite DB."""
    import optuna
    import pandas as pd

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    storage = f"sqlite:///{db_path}"
    rows = []
    for summary in optuna.study.get_all_study_summaries(storage=storage):
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        rows.append({
            "study": summary.study_name,
            "robust_best_score": study.user_attrs.get("robust_best_score"),
            "robust_best_params": study.user_attrs.get("robust_best_params"),
            "incumbent_score": study.user_attrs.get("incumbent_score"),
            "incumbent_params": study.user_attrs.get("incumbent_params"),
        })

    return pd.DataFrame(rows)
