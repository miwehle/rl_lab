"""Evaluation settings for HPO trial scores."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, kw_only=True)
class ScoringConfig:
    eval_episodes: int = 20
    eval_seed: int = 10_000

    def __post_init__(self) -> None:
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be >= 1")

    def study_attrs(self) -> dict:
        attrs = asdict(self)
        attrs["eval_seeds"] = list(
            range(self.eval_seed, self.eval_seed + self.eval_episodes)
        )
        del attrs["eval_seed"]
        return attrs
