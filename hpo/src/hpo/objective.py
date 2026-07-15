"""Shared Optuna objective for VectorTrainer DQN tasks."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import torch

from dqn.vector_training import VectorTrainer, VectorTrainingConfig, VectorTrainingResult
from hpo._logging import log_call
from hpo.hyperparams import HP

logger = logging.getLogger(__name__)


class _SuggestParameterValues(Protocol):
    def __call__(self, trial: Any, incumbent_params: dict[str, Any]) -> None: ...


class _EnvironmentFactory(Protocol):
    def make_training_env(self, num_envs: int, *, params: dict[str, Any]) -> Any: ...

    def evaluation_envs(self) -> dict[str, Callable[[], Any]]: ...


@dataclass
class ObjectiveContext:
    trial: Any
    params: dict[str, Any]
    training_config: VectorTrainingConfig
    trainer: Any | None = None
    training_result: VectorTrainingResult | None = None
    q_net: Any | None = None
    world_scores: dict[str, float] = field(default_factory=dict)
    score: float | None = None
    wall_time_seconds: float | None = None
    trial_seed: int | None = None


# We use the Hook Object pattern to keep objective() simple.
# It lets us plug in extensions and customizations
# without making objective.py know their concrete features.
#
# References:
#   Hook concept:
#     https://stackoverflow.com/questions/467557/what-is-meant-by-the-term-hook-in-programming
#   Dependency Injection:
#     https://martinfowler.com/articles/injection.html


class _Hooks(Protocol):
    def make_trainer(
        self, env, *, seed: int | None, device: Any, replay_memory_capacity: int, hidden_size: int
    ) -> Any: ...

    def q_net_for_evaluation(self, ctx: ObjectiveContext) -> Any: ...

    def training_plotter(self) -> Any | None: ...

    def finalize_trial(self, ctx: ObjectiveContext) -> None: ...


class _HookFactory(Protocol):
    def for_trial(self, ctx: ObjectiveContext) -> _Hooks: ...

    def study_attrs(self) -> dict[str, Any]: ...


class _DefaultHooks:
    def make_trainer(
        self, env, *, seed: int | None, device: Any, replay_memory_capacity: int, hidden_size: int
    ) -> Any:
        return VectorTrainer(
            env,
            seed=seed,
            device=device,
            replay_memory_capacity=replay_memory_capacity,
            hidden_size=hidden_size,
        )

    def q_net_for_evaluation(self, ctx: ObjectiveContext) -> Any:
        return ctx.training_result.q_net

    def training_plotter(self) -> Any | None:
        return None

    def finalize_trial(self, ctx: ObjectiveContext) -> None:
        pass


class _DefaultHookFactory:
    def for_trial(self, ctx: ObjectiveContext) -> _Hooks:
        return _DefaultHooks()

    def study_attrs(self) -> dict[str, Any]:
        return {}


@dataclass(frozen=True, kw_only=True)
class ObjectiveConfig:
    """Configure one Optuna objective.

    num_envs sets how many parallel training environments each trial uses.
    Internally, this is passed to VectorTrainer.
    """

    environment_factory: _EnvironmentFactory
    num_envs: int
    eval_episodes: int
    eval_max_steps: int = 2_000
    eval_seed: int = 10_000
    training_seed: int | None = 42
    early_stopping_score: float | None = -250.0
    hooks: _HookFactory = field(default_factory=_DefaultHookFactory)
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
            "early_stopping_score": self.early_stopping_score,
        }
        if self.device is not None:
            attrs["device"] = str(self.device)
        else:
            del attrs["device"]
        attrs.update(self.hooks.study_attrs())
        attrs["eval_seeds"] = list(range(self.eval_seed, self.eval_seed + self.eval_episodes))
        return attrs


def create_objective(
    *,
    suggest_parameter_values: _SuggestParameterValues,
    incumbent_params: dict[str, Any],
    config: ObjectiveConfig,
) -> Callable[[Any], float]:
    """Create an Optuna objective for one vectorized DQN trial."""

    @log_call
    def objective(trial: Any) -> float:
        # set up
        suggest_parameter_values(trial, incumbent_params)
        params = incumbent_params | trial.params
        training_config = _vector_training_config(params, early_stopping_score=config.early_stopping_score)
        ctx = ObjectiveContext(trial=trial, params=params, training_config=training_config)
        replay_memory_capacity = params[HP.REPLAY_MEMORY]
        trial_seed = None if config.training_seed is None else config.training_seed + trial.number
        ctx.trial_seed = trial_seed
        hooks = config.hooks.for_trial(ctx)
        env = config.environment_factory.make_training_env(config.num_envs, params=params)

        try:
            # make trainer
            ctx.trainer = hooks.make_trainer(
                env,
                seed=trial_seed,
                device=config.device,
                replay_memory_capacity=replay_memory_capacity,
                hidden_size=training_config.hidden_size,
            )

            # train DQN
            start_time = perf_counter()
            logger.info("VectorTrainer.train")
            ctx.training_result = ctx.trainer.train(training_config, plotter=hooks.training_plotter())
            ctx.wall_time_seconds = perf_counter() - start_time
        finally:
            env.close()

        if ctx.training_result.early_stopped:
            ctx.score = ctx.training_result.early_stopping_score
        else:
            ctx.q_net = hooks.q_net_for_evaluation(ctx)

            # evaluate DQN
            ctx.world_scores = {
                name: evaluate_greedy_q_net(
                    q_net=ctx.q_net,
                    device=ctx.trainer.device,
                    make_env=make_env,
                    episodes=config.eval_episodes,
                    max_steps=config.eval_max_steps,
                    seed=config.eval_seed,
                )
                for name, make_env in config.environment_factory.evaluation_envs().items()
            }
            ctx.score = sum(ctx.world_scores.values()) / len(ctx.world_scores)

        _set_user_attrs(ctx)
        hooks.finalize_trial(ctx)

        return ctx.score

    return objective


def _set_user_attrs(ctx: ObjectiveContext) -> None:
    def save(key, value):
        ctx.trial.set_user_attr(key, value)

    result = ctx.training_result
    if len(ctx.world_scores) > 1:
        save("world_scores", ctx.world_scores)
    save("env_steps", result.env_steps)
    save("optimizer_updates", result.optimizer_updates)
    save("trained_episodes", len(result.episode_returns))
    save("early_stopped", result.early_stopped)
    if result.early_stopping_score is not None:
        save("early_stopping_score", result.early_stopping_score)
    save("trial_seed", ctx.trial_seed)
    save("wall_time_seconds", ctx.wall_time_seconds)
    save(
        "training_curve",
        {
            "episode_returns": result.episode_returns,
            "episode_epsilons": result.episode_epsilons,
            "episode_env_indices": result.episode_env_indices,
        },
    )


def _vector_training_config(
    params: dict[str, Any], *, early_stopping_score: float | None = None
) -> VectorTrainingConfig:
    return VectorTrainingConfig(
        num_episodes=params[HP.NUM_EPISODES],
        batch_size=params[HP.BATCH_SIZE],
        gamma=params[HP.GAMMA],
        eps_start=params[HP.EPS_START],
        eps_end=params[HP.EPS_END],
        eps_decay=params[HP.EPS_DECAY],
        tau=params[HP.TAU],
        learning_rate=params[HP.LEARNING_RATE],
        learning_starts=params[HP.LEARNING_STARTS],
        optimize_every=params[HP.OPTIMIZE_EVERY],
        hidden_size=params.get(HP.HIDDEN_SIZE, 128),
        adaptive_extension_window=50,
        early_stopping_score=early_stopping_score,
    )


@log_call
def evaluate_greedy_q_net(
    *,
    q_net: Any,
    device,
    make_env: Callable[[], Any],
    episodes: int = 20,
    max_steps: int = 2_000,
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
                state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
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
