"""Shared Optuna objective for VectorTrainer DQN tasks."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import torch

from dqn.vector_training import VectorTrainer, VectorTrainingConfig
from hpo.lunar_lander.logging import log_call
from hpo.hyperparams import HP


logger = logging.getLogger(__name__)


class SuggestParameterValues(Protocol):
    def __call__(self, trial: Any, incumbent_params: dict[str, Any]) -> None: ...


class EnvironmentFactory(Protocol):
    def make_training_env(self, num_envs: int) -> Any: ...

    def evaluation_envs(self) -> dict[str, Callable[[], Any]]: ...


# We use the Hook Object pattern to keep objective() simple.
# It lets us plug in extensions and customizations
# without making objective.py know their concrete features.
#
# References:
#   Hook concept:
#     https://stackoverflow.com/questions/467557/what-is-meant-by-the-term-hook-in-programming
#   Dependency Injection:
#     https://martinfowler.com/articles/injection.html

class Hooks(Protocol):
    def make_trainer(
        self,
        env,
        *,
        seed: int | None,
        device: Any,
        replay_memory_capacity: int,
    ) -> Any: ...

    def q_net_for_evaluation(self, q_net: Any, device: Any) -> Any: ...

    def save_trial_attrs(self, save: Callable[[str, Any], None]) -> None: ...


class HookFactory(Protocol):
    def for_trial(self, trial: Any, training_config: VectorTrainingConfig) -> Hooks: ...

    def study_attrs(self) -> dict[str, Any]: ...


class DefaultHooks:
    def make_trainer(
        self,
        env,
        *,
        seed: int | None,
        device: Any,
        replay_memory_capacity: int,
    ) -> Any:
        return VectorTrainer(
            env,
            seed=seed,
            device=device,
            replay_memory_capacity=replay_memory_capacity,
        )

    def q_net_for_evaluation(self, q_net: Any, device: Any) -> Any:
        return q_net

    def save_trial_attrs(self, save: Callable[[str, Any], None]) -> None:
        pass


class DefaultHookFactory:
    def for_trial(self, trial: Any, training_config: VectorTrainingConfig) -> Hooks:
        return DefaultHooks()

    def study_attrs(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True, kw_only=True)
class ObjectiveConfig:    
    """Configure one Optuna objective.

    num_envs sets how many parallel training environments each trial uses.
    Internally, this is passed to VectorTrainer.
    """
    environment_factory: EnvironmentFactory
    num_envs: int
    eval_episodes: int
    eval_max_steps: int = 2_000
    eval_seed: int = 10_000
    training_seed: int | None = 42
    hooks: HookFactory = field(default_factory=DefaultHookFactory)
    device: Any = None

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be >= 1")
        if self.eval_max_steps < 1:
            raise ValueError("eval_max_steps must be >= 1")

    def study_attrs(self) -> dict:
        attrs = {
            "num_envs": self.num_envs,
            "device": self.device,
            "eval_episodes": self.eval_episodes,
            "eval_max_steps": self.eval_max_steps,
            "training_seed": self.training_seed,
        }
        if self.device is not None:
            attrs["device"] = str(self.device)
        else:
            del attrs["device"]
        attrs.update(self.hooks.study_attrs())
        attrs["eval_seeds"] = list(
            range(self.eval_seed, self.eval_seed + self.eval_episodes)
        )
        return attrs


def create_objective(
    *, suggest_parameter_values: SuggestParameterValues,
    incumbent_params: dict[str, Any],
    config: ObjectiveConfig,
) -> Callable[[Any], float]:
    """Create an Optuna objective for one vectorized DQN trial."""

    @log_call
    def objective(trial: Any) -> float:
        suggest_parameter_values(trial, incumbent_params)
        params = incumbent_params | trial.params
        training_config = vector_training_config(params)
        replay_memory_capacity = params[HP.REPLAY_MEMORY_CAPACITY]
        trial_seed = (
            None
            if config.training_seed is None
            else config.training_seed + trial.number
        )
        hooks = config.hooks.for_trial(trial, training_config)

        env = config.environment_factory.make_training_env(config.num_envs)
        try:
            trainer = hooks.make_trainer(
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

        q_net = hooks.q_net_for_evaluation(result.q_net, trainer.device)

        world_scores = {
            name: evaluate_greedy_q_net(
                q_net=q_net,
                device=trainer.device,
                make_env=make_env,
                episodes=config.eval_episodes,
                max_steps=config.eval_max_steps,
                seed=config.eval_seed,
            )
            for name, make_env in config.environment_factory.evaluation_envs().items()
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
        hooks.save_trial_attrs(save)
        save("training_curve", {
            "episode_returns": result.episode_returns,
            "episode_epsilons": result.episode_epsilons,
        })

        return score

    return objective


def vector_training_config(params: dict[str, Any]) -> VectorTrainingConfig:
    return VectorTrainingConfig(
        num_episodes=params[HP.NUM_EPISODES],
        batch_size=params[HP.BATCH_SIZE],
        gamma=params[HP.GAMMA],
        eps_start=1.0,
        eps_end=params[HP.EPS_END],
        eps_decay=params[HP.EPS_DECAY],
        tau=params[HP.TAU],
        learning_rate=params[HP.LEARNING_RATE],
        learning_starts=params[HP.LEARNING_STARTS],
        optimize_every=params[HP.OPTIMIZE_EVERY],
    )


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
