import csv

import yaml

from reward_shaping.experiment_harness import (
    EvaluationResult,
    EvaluationRow,
    prepare_run_directory,
    write_eval_scores,
    write_yaml,
)


def test_prepare_run_directory_copies_initial_checkpoint_and_returns_standard_paths(tmp_path) -> None:
    initial_checkpoint = tmp_path / "source.pt"
    initial_checkpoint.write_text("checkpoint", encoding="utf-8")

    paths = prepare_run_directory(tmp_path / "runs", "run-1", initial_checkpoint=initial_checkpoint)

    assert paths.initial_checkpoint.read_text(encoding="utf-8") == "checkpoint"
    assert paths.config == tmp_path / "runs" / "run-1" / "outputs" / "config.yaml"
    assert paths.shaped_checkpoint == tmp_path / "runs" / "run-1" / "outputs" / "shaped_checkpoint.pt"


def test_write_yaml_writes_human_readable_mapping(tmp_path) -> None:
    path = tmp_path / "config.yaml"

    write_yaml(path, {"run_id": "run-1", "penalty": 0.5})

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"run_id": "run-1", "penalty": 0.5}


def test_write_eval_scores_writes_rows_for_all_measurements(tmp_path) -> None:
    path = tmp_path / "eval_scores.csv"
    result = EvaluationResult(
        measurement="historical_score",
        score=1.0,
        world_scores={"earth": 1.0},
        episodes_per_world=1,
        eval_seed=10_000,
        rows=[
            EvaluationRow(
                measurement="historical_score",
                world="earth",
                episode=0,
                seed=10_000,
                score=1.0,
                ground_side_thrust_steps=2,
            )
        ],
    )

    write_eval_scores(path, [result])

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows == [
        {
            "measurement": "historical_score",
            "world": "earth",
            "episode": "0",
            "seed": "10000",
            "score": "1.0",
            "ground_side_thrust_steps": "2",
        }
    ]
