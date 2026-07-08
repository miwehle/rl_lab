"""Run one reward shaping experiment."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dqn.vector_training import VectorTrainer, VectorTrainingConfig
from reward_shaping import make_reward_shaping_vector_env
from reward_shaping.experiment_harness.artifacts import (
    RunPaths,
    prepare_run_directory,
    write_eval_scores,
    write_yaml,
)
from reward_shaping.experiment_harness.checkpointing import load_q_net_checkpoint, save_q_net_checkpoint
from reward_shaping.experiment_harness.evaluation import EvaluationResult, historical_score, robust_score


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    paths: RunPaths
    q_net: Any
    mean_return: float
    historical_score: EvaluationResult
    robust_score: EvaluationResult


@dataclass(frozen=True)
class ExperimentContext:
    source_storage_name: str
    drive_study_dir: Path
    training_factory: Any
    evaluation_factory: Any
    replay_memory_capacity: int
    ground_thrust_penalty: float
    num_envs: int
    robust_episodes_per_world: int
    training_seed: int
    device: Any = None
    run_root: Path = Path("/content/rl_lab/reward_shaping/runs")
    drive_run_root: Path = Path("/content/drive/MyDrive/rl_lab/reward_shaping/runs")

    @property
    def initial_checkpoint(self) -> Path:
        return (
            self.drive_study_dir
            / "best_checkpoints"
            / self.source_storage_name
            / "best_eval_checkpoint.pt"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source_storage_name": self.source_storage_name,
            "observation_mode": self.training_factory.observation_mode,
            "training_world_mix": _world_mix(self.training_factory),
            "initial_checkpoint": str(self.initial_checkpoint),
            "run_root": str(self.run_root),
            "drive_run_root": str(self.drive_run_root),
            "ground_thrust_penalty": self.ground_thrust_penalty,
            "num_envs": self.num_envs,
            "training_seed": self.training_seed,
            "replay_memory_capacity": self.replay_memory_capacity,
            "historical_score": {"eval_seed": 10_000, "episodes_per_world": 10},
            "robust_score": {"eval_seed": 10_000, "episodes_per_world": self.robust_episodes_per_world},
        }


def run_experiment(
    context: ExperimentContext,
    *,
    training_config: VectorTrainingConfig,
) -> ExperimentResult:
    run_id = _run_id(context, training_config)
    paths = prepare_run_directory(context.run_root, run_id, initial_checkpoint=context.initial_checkpoint)

    def train() -> tuple[Any, float]:
        training_env = make_reward_shaping_vector_env(
            context.training_factory,
            context.num_envs,
            ground_thrust_penalty=context.ground_thrust_penalty,
        )
        try:
            trainer = VectorTrainer(
                training_env,
                seed=context.training_seed,
                device=context.device,
                replay_memory_capacity=context.replay_memory_capacity,
                hidden_size=training_config.hidden_size,
            )
            load_q_net_checkpoint(trainer.q_net, paths.initial_checkpoint, context.device)
            trainer.target_net.load_state_dict(trainer.q_net.state_dict())
            training_result = trainer.train(training_config)
        finally:
            training_env.close()

        mean_return = _mean(training_result.episode_returns)
        write_yaml(paths.training_summary, {"mean_return": mean_return})
        return trainer.q_net, mean_return

    def evaluate(q_net) -> tuple[EvaluationResult, EvaluationResult]:
        make_envs = context.evaluation_factory.evaluation_envs()
        return (
            historical_score(q_net=q_net, make_envs=make_envs, device=context.device),
            robust_score(
                q_net=q_net,
                make_envs=make_envs,
                episodes_per_world=context.robust_episodes_per_world,
                device=context.device,
            ),
        )

    write_yaml(
        paths.config,
        context.metadata() | {"run_id": run_id, "training_config": training_config.__dict__},
    )

    q_net, mean_return = train()
    historical, robust = evaluate(q_net)
    write_eval_scores(paths.eval_scores, [historical, robust])
    save_q_net_checkpoint(
        q_net,
        paths.shaped_checkpoint,
        {
            "run_id": run_id,
            "mean_return": mean_return,
            "historical_score": historical.score,
            "robust_score": robust.score,
        },
    )
    return ExperimentResult(run_id, paths, q_net, mean_return, historical, robust)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _run_id(context: ExperimentContext, training_config: VectorTrainingConfig) -> str:
    return (
        "ground_thrust_penalty"
        f"_ep{training_config.num_episodes}"
        f"_penalty{_value_label(context.ground_thrust_penalty)}"
        f"_eps{_value_label(training_config.eps_start)}"
    )


def _value_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _world_mix(factory) -> dict[str, int]:
    return dict(Counter(world.name for world in factory.worlds))
