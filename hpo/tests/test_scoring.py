import pytest

from hpo.evaluation.scoring import best_window_mean


def test_best_window_mean_uses_best_rolling_window() -> None:
    score = best_window_mean([1.0, 8.0, 6.0, 2.0], 2)

    assert score.mean == pytest.approx(7.0)
    assert score.start_episode == 2
    assert score.end_episode == 3


def test_best_window_mean_uses_available_values_when_window_is_larger() -> None:
    score = best_window_mean([1.0, 3.0], 5)

    assert score.mean == pytest.approx(2.0)
    assert score.start_episode == 1
    assert score.end_episode == 2


def test_best_window_mean_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="values must not be empty"):
        best_window_mean([], 5)
