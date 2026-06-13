"""Optional Optuna pruning for HPO trials."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dqn.tuned_training import AfterEpisodeCallback
from hpo.evaluation.scoring import best_window_mean


@dataclass(frozen=True, kw_only=True)
class PruningConfig:
    start_episode: int = 250
    min_score: float = 100.0

    def __post_init__(self) -> None:
        if self.start_episode < 1:
            raise ValueError("start_episode must be >= 1")


def create_pruning_callback(
    trial: Any,
    config: PruningConfig | None,
    *,
    num_episodes: int,
    score_window: int,
) -> AfterEpisodeCallback | None:
    """Create an after-episode callback that reports and optionally prunes."""
    if config is None:
        return None

    def callback(episode_returns: Sequence[float]) -> None:
        episode = len(episode_returns)
        if episode < config.start_episode:
            return

        score = best_window_mean(episode_returns, score_window)
        trial.report(score.mean, step=episode)

        if score.mean < config.min_score:
            trial.set_user_attr("pruned_at_episode", episode)
            trial.set_user_attr("episodes_saved_by_pruning", num_episodes - episode)
            raise _trial_pruned(
                f"best window mean {score.mean:.1f} below {config.min_score:.1f} "
                f"at episode {episode}"
            )

    return callback


def _trial_pruned(message: str) -> Exception:
    import optuna

    return optuna.TrialPruned(message)
