import pytest
import torch

from dqn.vector_training import VectorTrainingConfig
from hpo.checkpointing import (
    BestCheckpointRecorder,
    ObjectiveHookFactory,
    load_checkpoint,
)


class FakeTrainer:
    def __init__(self) -> None:
        self.q_net = torch.nn.Linear(1, 1)


class FakeTrial:
    number = 3


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


def test_best_checkpoint_recorder_saves_best_full_window(tmp_path) -> None:
    path = tmp_path / "best.pt"
    trainer = FakeTrainer()
    recorder = BestCheckpointRecorder(
        path,
        window=2,
        metadata={"trial_number": 7},
    )

    recorder.after_episode(trainer, [1.0])

    assert not path.exists()

    set_weights(trainer.q_net, 2.0)
    recorder.after_episode(trainer, [1.0, 3.0])

    assert path.exists()
    assert recorder.best_score == pytest.approx(2.0)
    assert recorder.best_episode == 2

    set_weights(trainer.q_net, 5.0)
    recorder.after_episode(trainer, [1.0, 3.0, 4.0])

    restored = torch.nn.Linear(1, 1)
    metadata = load_checkpoint(restored, path, torch.device("cpu"))

    assert first_weight(restored) == pytest.approx(5.0)
    assert metadata["trial_number"] == 7
    assert metadata["score"] == pytest.approx(3.5)
    assert metadata["episode"] == 3
    assert metadata["window"] == 2


def test_checkpointing_objective_hook_factory_reports_study_attrs(tmp_path) -> None:
    factory = ObjectiveHookFactory(
        tmp_path,
        window=50,
        min_score=100.0,
        min_score_delta=5.0,
    )

    assert factory.study_attrs() == {
        "checkpoint_dir": str(tmp_path),
        "checkpoint_window": 50,
        "checkpoint_min_score": 100.0,
        "checkpoint_min_score_delta": 5.0,
    }


def test_checkpointing_objective_hooks_load_best_checkpoint_and_save_attrs(
    tmp_path,
) -> None:
    hooks = ObjectiveHookFactory(
        tmp_path,
        window=2,
    ).for_trial(FakeTrial(), training_config())
    trainer = FakeTrainer()

    set_weights(trainer.q_net, 7.0)
    hooks.recorder.after_episode(trainer, [1.0, 3.0])
    set_weights(trainer.q_net, 1.0)

    q_net = hooks.q_net_for_evaluation(trainer.q_net, torch.device("cpu"))
    assert q_net is trainer.q_net
    assert first_weight(trainer.q_net) == pytest.approx(7.0)

    attrs = {}
    hooks.save_trial_attrs(attrs.__setitem__)

    assert attrs["checkpoint_path"] == str(tmp_path / "trial_0003_best.pt")
    assert attrs["checkpoint_score"] == pytest.approx(2.0)
    assert attrs["checkpoint_episode"] == 2
    assert attrs["checkpoint_window"] == 2
