from types import SimpleNamespace

import pytest

from dqn.vector_training import VectorTrainingConfig
from reward_shaping.experiment_harness import EvaluationResult, ExperimentContext
from reward_shaping.experiment_harness import runner as runner_module


class FakeEnv:
    closed = False

    def close(self) -> None:
        self.closed = True


class FakeTargetNet:
    def load_state_dict(self, state) -> None:
        self.state = state


class FakeQNet:
    def state_dict(self):
        return {"weight": 1.0}


class FakeWorld:
    name = "moon"


class FakeTrainingFactory:
    observation_mode = "10d"
    worlds = (FakeWorld(),)


class FakeTrainer:
    def __init__(self, env, *, seed, device, replay_memory_capacity, hidden_size) -> None:
        self.env = env
        self.seed = seed
        self.device = device
        self.replay_memory_capacity = replay_memory_capacity
        self.hidden_size = hidden_size
        self.q_net = FakeQNet()
        self.target_net = FakeTargetNet()

    def train(self, config):
        return SimpleNamespace(episode_returns=[1.0, 3.0], env_steps=2, optimizer_updates=1)


def test_run_experiment_writes_artifacts_and_returns_scores(tmp_path, monkeypatch) -> None:
    training_env = FakeEnv()
    saved_checkpoints = []
    drive_study_dir = tmp_path / "drive-study"
    initial_checkpoint = drive_study_dir / "best_checkpoints" / "study-1" / "best_eval_checkpoint.pt"
    initial_checkpoint.parent.mkdir(parents=True)
    initial_checkpoint.write_text("checkpoint", encoding="utf-8")
    context = ExperimentContext(
        source_storage_name="study-1",
        drive_study_dir=drive_study_dir,
        run_root=tmp_path / "runs",
        drive_run_root=tmp_path / "drive-runs",
        training_factory=FakeTrainingFactory(),
        evaluation_factory=SimpleNamespace(evaluation_envs=lambda: {}),
        replay_memory_capacity=100,
        ground_thrust_penalty=0.5,
        num_envs=2,
        robust_episodes_per_world=3,
        training_seed=77,
        device="cpu",
    )

    monkeypatch.setattr(runner_module, "make_reward_shaping_vector_env", lambda *args, **kwargs: training_env)
    monkeypatch.setattr(runner_module, "VectorTrainer", FakeTrainer)
    monkeypatch.setattr(runner_module, "load_q_net_checkpoint", lambda *args, **kwargs: {"score": 253.0})
    monkeypatch.setattr(
        runner_module,
        "historical_score",
        lambda **kwargs: _evaluation_result("historical_score", 10.0),
    )
    monkeypatch.setattr(
        runner_module,
        "robust_score",
        lambda **kwargs: _evaluation_result("robust_score", 20.0),
    )
    monkeypatch.setattr(
        runner_module,
        "save_q_net_checkpoint",
        lambda q_net, path, metadata: saved_checkpoints.append((path, metadata)) or path.write_text("saved"),
    )

    result = runner_module.run_experiment(
        context,
        training_config=_training_config(),
    )

    assert result.run_id == "ground_thrust_penalty_ep2_penalty0p5_eps0p1"
    assert training_env.closed
    assert result.q_net.state_dict() == {"weight": 1.0}
    assert result.mean_return == pytest.approx(2.0)
    assert result.historical_score.score == pytest.approx(10.0)
    assert result.robust_score.score == pytest.approx(20.0)
    assert result.paths.initial_checkpoint.read_text(encoding="utf-8") == "checkpoint"
    config = result.paths.config.read_text(encoding="utf-8")
    assert "run_id: ground_thrust_penalty_ep2_penalty0p5_eps0p1" in config
    assert "observation_mode: 10d" in config
    assert "training_world_mix:" in config
    assert result.paths.training_summary.read_text(encoding="utf-8") == "mean_return: 2.0\n"
    assert result.paths.eval_scores.exists()
    assert result.paths.shaped_checkpoint.read_text() == "saved"
    assert saved_checkpoints[0][0] == result.paths.shaped_checkpoint
    assert saved_checkpoints[0][1]["mean_return"] == pytest.approx(2.0)
    assert saved_checkpoints[0][1]["historical_score"] == pytest.approx(10.0)
    assert saved_checkpoints[0][1]["robust_score"] == pytest.approx(20.0)


def _training_config() -> VectorTrainingConfig:
    return VectorTrainingConfig(
        num_episodes=2,
        batch_size=1,
        eps_start=0.1,
        eps_end=0.1,
        eps_decay=10,
        learning_rate=0.001,
        gamma=0.99,
        tau=0.01,
        learning_starts=0,
        optimize_every=1,
        hidden_size=4,
    )


def _evaluation_result(measurement: str, score: float) -> EvaluationResult:
    return EvaluationResult(
        measurement=measurement,
        score=score,
        world_scores={"earth": score},
        episodes_per_world=1,
        eval_seed=10_000,
        rows=[],
    )
