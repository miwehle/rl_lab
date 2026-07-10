import json

import pytest
import torch

from dqn.vector_training import VectorTrainingConfig
from hpo.checkpointing import (
    BestCheckpointRecorder,
    EvaluationBestCheckpointRecorder,
    ObjectiveHookFactory,
    best_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from hpo.objective import ObjectiveContext


class FakeTrainer:
    def __init__(self) -> None:
        self.q_net = torch.nn.Linear(1, 1)


class FakeTrial:
    number = 3
    params = {"learning_rate": 0.002}

    def __init__(self) -> None:
        self.user_attrs = {}

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


class FakeRobustTrial(FakeTrial):
    number = 3
    checkpoint_subdir = "robustness"
    checkpoint_stem = "trial_0003_seed_1001"


class FakeStudy:
    def __init__(self, checkpoint_dir) -> None:
        self.user_attrs = {"checkpoint_dir": str(checkpoint_dir)}


def training_config() -> VectorTrainingConfig:
    return VectorTrainingConfig(
        num_episodes=12,
        batch_size=64,
        eps_start=0.7,
        eps_end=0.02,
        eps_decay=1234,
        learning_rate=5e-4,
        learning_starts=77,
        optimize_every=3,
    )


def set_weights(model: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def first_weight(model: torch.nn.Module) -> float:
    return next(model.parameters()).flatten()[0].item()


def linear_with_weight(value: float) -> torch.nn.Linear:
    model = torch.nn.Linear(1, 1)
    set_weights(model, value)
    return model


def loaded_weight(path, *, device=torch.device("cpu")) -> tuple[float, dict]:
    restored = torch.nn.Linear(1, 1)
    metadata = load_checkpoint(restored, path, device)
    return first_weight(restored), metadata


def archive_hooks(tmp_path, archive_dir):
    return ObjectiveHookFactory(tmp_path / "checkpoints", best_eval_archive_dir=archive_dir).for_trial(
        objective_context()
    )


def objective_context(
    trial=None, config=None, *, params=None, q_net=None, score=None, episode_returns=None
) -> ObjectiveContext:
    ctx = ObjectiveContext(
        trial=trial or FakeTrial(),
        params=params or {"learning_rate": 0.002, "gamma": 0.99},
        training_config=config or training_config(),
    )
    if q_net is not None:
        ctx.q_net = q_net
    if score is not None:
        ctx.score = score
    if episode_returns is not None:
        from dqn.vector_training import VectorTrainingResult

        ctx.training_result = VectorTrainingResult(
            q_net=q_net,
            episode_returns=episode_returns,
            episode_lengths=[1] * len(episode_returns),
            episode_epsilons=[0.1] * len(episode_returns),
            episode_env_indices=[0] * len(episode_returns),
            env_steps=len(episode_returns),
            optimizer_updates=0,
        )
    return ctx


def test_best_checkpoint_recorder_saves_best_full_window(tmp_path) -> None:
    path = tmp_path / "best.pt"
    trainer = FakeTrainer()
    recorder = BestCheckpointRecorder(path, window=2, metadata={"trial_number": 7})

    recorder.after_episode(trainer, [1.0])

    assert not path.exists()

    set_weights(trainer.q_net, 2.0)
    recorder.after_episode(trainer, [1.0, 3.0])

    assert path.exists()
    assert recorder.best_score == pytest.approx(2.0)
    assert recorder.best_episode == 2

    set_weights(trainer.q_net, 5.0)
    recorder.after_episode(trainer, [1.0, 3.0, 4.0])

    weight, metadata = loaded_weight(path)

    assert weight == pytest.approx(5.0)
    assert metadata["trial_number"] == 7
    assert metadata["score"] == pytest.approx(3.5)
    assert metadata["episode"] == 3
    assert metadata["window"] == 2


def test_checkpointing_objective_hook_factory_reports_study_attrs(tmp_path) -> None:
    factory = ObjectiveHookFactory(
        tmp_path, window=50, min_score=100.0, min_score_delta=5.0, best_eval_archive_dir=tmp_path / "archive"
    )

    assert factory.study_attrs() == {
        "checkpoint_dir": str(tmp_path),
        "checkpoint_window": 50,
        "checkpoint_min_score": 100.0,
        "checkpoint_min_score_delta": 5.0,
        "best_eval_archive_dir": str(tmp_path / "archive"),
    }


def test_checkpointing_objective_hook_factory_copies_with_min_score(tmp_path) -> None:
    factory = ObjectiveHookFactory(tmp_path, window=50, min_score=100.0)

    copied = factory.with_min_score(120.0)

    assert copied.min_score == pytest.approx(120.0)
    assert factory.min_score == pytest.approx(100.0)
    assert copied.with_min_score(90.0).min_score == pytest.approx(120.0)


def test_checkpointing_objective_hooks_create_training_plotter(tmp_path) -> None:
    progress_calls = []
    hooks = (
        ObjectiveHookFactory(tmp_path, window=2, min_score=10.0)
        .with_training_progress(progress_calls.append)
        .for_trial(objective_context())
    )
    trainer = FakeTrainer()

    plotter = hooks.training_plotter()
    assert plotter is not None

    plotter.plot_returns([1.0])
    hooks.recorder.after_episode(trainer, [10.0, 20.0])
    plotter.plot_returns([10.0, 20.0])

    assert progress_calls[0].checkpoint_window == 2
    assert progress_calls[0].checkpoint_min_score == pytest.approx(10.0)
    assert progress_calls[0].trial_params == {"learning_rate": 0.002, "gamma": 0.99}
    assert progress_calls[0].optimized_param_names == ["learning_rate"]
    assert progress_calls[0].best_checkpoint_score is None
    assert progress_calls[1].best_checkpoint_score == pytest.approx(15.0)


def test_checkpointing_objective_hooks_load_best_checkpoint_and_save_attrs(tmp_path) -> None:
    hooks = ObjectiveHookFactory(tmp_path, window=2).for_trial(objective_context())
    trainer = FakeTrainer()

    set_weights(trainer.q_net, 7.0)
    hooks.recorder.after_episode(trainer, [1.0, 3.0])
    set_weights(trainer.q_net, 1.0)

    assert hooks.checkpoint_window == 2
    assert hooks.best_checkpoint_score == pytest.approx(2.0)

    ctx = objective_context(q_net=trainer.q_net, episode_returns=[1.0, 3.0])
    q_net = hooks.q_net_for_evaluation(ctx)
    assert q_net is trainer.q_net
    assert first_weight(trainer.q_net) == pytest.approx(7.0)

    hooks.finalize_trial(ctx)
    attrs = ctx.trial.user_attrs

    assert attrs["checkpoint_path"] == str(tmp_path / "trials" / "trial_0003_best.pt")
    assert attrs["checkpoint_score"] == pytest.approx(2.0)
    assert attrs["checkpoint_episode"] == 2
    assert attrs["checkpoint_window"] == 2


def test_checkpointing_objective_hook_factory_uses_robustness_checkpoint_dir(tmp_path) -> None:
    hooks = ObjectiveHookFactory(tmp_path, window=2).for_trial(objective_context(trial=FakeRobustTrial()))
    trainer = FakeTrainer()

    hooks.recorder.after_episode(trainer, [1.0, 3.0])

    ctx = objective_context(trial=FakeRobustTrial())
    hooks.finalize_trial(ctx)
    attrs = ctx.trial.user_attrs

    assert attrs["checkpoint_path"] == str(tmp_path / "robustness" / "trial_0003_seed_1001_best.pt")


def test_evaluation_best_checkpoint_recorder_saves_evaluated_model(tmp_path) -> None:
    path = tmp_path / "eval_best.pt"
    ctx = objective_context(q_net=linear_with_weight(9.0), score=211.0, episode_returns=[1.0, 2.0, 3.0])
    ctx.world_scores = {"moon": 211.0}
    recorder = EvaluationBestCheckpointRecorder(path, min_score=200.0)

    recorder.after_evaluation(ctx)

    weight, metadata = loaded_weight(path)
    assert weight == pytest.approx(9.0)
    assert metadata["score"] == pytest.approx(211.0)
    assert metadata["episode"] == 3
    assert metadata["window"] is None


def test_objective_hooks_archive_best_eval_checkpoint(tmp_path) -> None:
    archive_dir = tmp_path / "archive"
    hooks = archive_hooks(tmp_path, archive_dir)
    ctx = objective_context(q_net=linear_with_weight(9.0), score=211.0, episode_returns=[1.0, 2.0, 3.0])

    hooks.finalize_trial(ctx)

    archive_path = archive_dir / "best_eval_checkpoint.pt"
    metadata = json.loads((archive_dir / "best_eval_checkpoint.json").read_text())
    weight, _ = loaded_weight(archive_path)

    assert ctx.trial.user_attrs["evaluation_checkpoint_archive_path"] == str(archive_path)
    assert weight == pytest.approx(9.0)
    assert metadata["score"] == pytest.approx(211.0)
    assert metadata["source_path"] == str(tmp_path / "checkpoints" / "trials" / "trial_0003_eval_best.pt")


def test_objective_hooks_keep_archived_eval_checkpoint_when_score_is_lower(tmp_path) -> None:
    archive_dir = tmp_path / "archive"
    high_hooks = archive_hooks(tmp_path, archive_dir)
    high_ctx = objective_context(q_net=linear_with_weight(9.0), score=211.0, episode_returns=[1.0])
    high_hooks.finalize_trial(high_ctx)

    low_hooks = archive_hooks(tmp_path, archive_dir)
    low_ctx = objective_context(q_net=linear_with_weight(3.0), score=200.0, episode_returns=[1.0])

    low_hooks.finalize_trial(low_ctx)

    weight, metadata = loaded_weight(archive_dir / "best_eval_checkpoint.pt")

    assert "evaluation_checkpoint_archive_path" not in low_ctx.trial.user_attrs
    assert weight == pytest.approx(9.0)
    assert metadata["score"] == pytest.approx(211.0)


def test_best_checkpoint_selects_highest_score_across_checkpoint_dirs(tmp_path) -> None:
    model = torch.nn.Linear(1, 1)
    trial_path = tmp_path / "trials" / "trial_0001_best.pt"
    robustness_path = tmp_path / "robustness" / "trial_0001_seed_1001_best.pt"

    save_checkpoint(model, trial_path, {"score": 10.0, "episode": 3, "window": 2})
    save_checkpoint(model, robustness_path, {"score": 20.0, "episode": 4, "window": 2})

    checkpoint = best_checkpoint(FakeStudy(tmp_path))

    assert checkpoint.path == robustness_path
    assert checkpoint.score == pytest.approx(20.0)
    assert checkpoint.episode == 4
    assert checkpoint.window == 2
    assert checkpoint.source == "robustness"


def test_best_checkpoint_prefers_evaluation_scores(tmp_path) -> None:
    model = torch.nn.Linear(1, 1)
    training_path = tmp_path / "trials" / "trial_0001_best.pt"
    evaluation_path = tmp_path / "trials" / "trial_0001_eval_best.pt"

    save_checkpoint(model, training_path, {"score": 250.0, "episode": 3, "window": 2})
    save_checkpoint(model, evaluation_path, {"score": 211.0, "episode": 4, "window": None})

    checkpoint = best_checkpoint(FakeStudy(tmp_path))

    assert checkpoint.path == evaluation_path
    assert checkpoint.score == pytest.approx(211.0)
    assert checkpoint.window is None


def test_best_checkpoint_rejects_study_without_checkpoints(tmp_path) -> None:
    with pytest.raises(ValueError, match="study has no checkpoints"):
        best_checkpoint(FakeStudy(tmp_path))
