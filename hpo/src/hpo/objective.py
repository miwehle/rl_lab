"""Shared Optuna objective for VectorTrainer DQN tasks."""

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Protocol

import torch

from dqn.vector_training import VectorTrainer, VectorTrainingConfig
from hpo.lunar_lander.logging import log_call


logger = logging.getLogger(__name__)


class SearchSpace(Protocol):
    def training_config(
        self, trial: Any, incumbent_params: dict[str, Any]
    ) -> VectorTrainingConfig: ...

    def replay_memory_capacity(
        self, trial: Any, incumbent_params: dict[str, Any]
    ) -> int: ...


class EnvironmentFactory(Protocol):
    def make_training_env(self, num_envs: int) -> Any: ...

    def evaluation_envs(self) -> dict[str, Callable[[], Any]]: ...


@dataclass(frozen=True, kw_only=True)
class ObjectiveConfig:
    num_envs: int = 16
    training_seed: int | None = 42
    device: Any = None
    eval_episodes: int = 20
    eval_seed: int = 10_000
    eval_max_steps: int = 2_000

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be >= 1")
        if self.eval_max_steps < 1:
            raise ValueError("eval_max_steps must be >= 1")

    def study_attrs(self) -> dict:
        attrs = asdict(self)
        attrs["eval_seeds"] = list(
            range(self.eval_seed, self.eval_seed + self.eval_episodes)
        )
        del attrs["eval_seed"]
        return attrs


def create_objective(
    *, search_space: SearchSpace,
    incumbent_params: dict[str, Any],
    environment_factory: EnvironmentFactory,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> Callable[[Any], float]:
    """Create an Optuna objective for one vectorized DQN trial."""

    @log_call
    def objective(trial: Any) -> float:
        training_config = search_space.training_config(trial, incumbent_params)
        replay_memory_capacity = search_space.replay_memory_capacity(
            trial, incumbent_params
        )
        trial_seed = (
            None
            if config.training_seed is None
            else config.training_seed + trial.number
        )

        env = environment_factory.make_training_env(config.num_envs)
        try:
            trainer = VectorTrainer(
                env,
                seed=trial_seed,
                device=config.device,
                replay_memory_capacity=replay_memory_capacity,
            )

            start_time = perf_counter()
            logger.info("VectorTrainer.train")
            result = trainer.train(training_config)
            wall_time_seconds = perf_counter() - start_time
        finally:
            env.close()

        world_scores = {
            name: evaluate_greedy_q_net(
                q_net=result.q_net,
                device=trainer.device,
                make_env=make_env,
                episodes=config.eval_episodes,
                max_steps=config.eval_max_steps,
                seed=config.eval_seed,
            )
            for name, make_env in environment_factory.evaluation_envs().items()
        }
        score = sum(world_scores.values()) / len(world_scores)

        def save(key, value):
            trial.set_user_attr(key, value)

        if len(world_scores) > 1:
            save("world_scores", world_scores)
        save("env_steps", result.env_steps)
        save("optimizer_updates", result.optimizer_updates)
        save("trial_seed", trial_seed)
        save("wall_time_seconds", wall_time_seconds)
        save("training_curve", {
            "episode_returns": result.episode_returns,
            "episode_epsilons": result.episode_epsilons,
        })

        return score

    return objective


@log_call
def evaluate_greedy_q_net(
    *, q_net: Any, device, make_env: Callable[[], Any],
    episodes: int = 20, max_steps: int = 2_000,
    seed: int | None = None,
) -> float:
    """Return mean greedy episode return for a trained Q-network."""
    episode_returns = []
    q_net.eval()

    for episode in range(episodes):
        env = make_env()
        try:
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)
            episode_return = 0.0

            for _ in range(max_steps):
                state = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    # Greedy action: choose the highest Q-value without exploration.
                    action = q_net(state).argmax(dim=1).item()

                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
        finally:
            env.close()

        episode_returns.append(episode_return)

    return sum(episode_returns) / len(episode_returns)
