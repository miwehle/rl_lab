import pytest

from hpo.evaluation.scoring import training_effort


def test_training_effort_combines_relative_env_and_learning_work() -> None:
    assert training_effort(
        env_steps=80,
        processed_samples=120,
        baseline_env_steps=100,
        baseline_processed_samples=100,
    ) == pytest.approx(1.0)


def test_training_effort_validates_baseline_and_alpha() -> None:
    with pytest.raises(ValueError, match="baseline"):
        training_effort(
            env_steps=1,
            processed_samples=1,
            baseline_env_steps=0,
            baseline_processed_samples=1,
        )
    with pytest.raises(ValueError, match="alpha"):
        training_effort(
            env_steps=1,
            processed_samples=1,
            baseline_env_steps=1,
            baseline_processed_samples=1,
            alpha=2,
        )
