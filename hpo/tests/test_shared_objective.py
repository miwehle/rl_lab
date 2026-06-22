from dataclasses import dataclass

import pytest
import torch

from dqn.vector_training import VectorTrainingConfig, VectorTrainingResult
from hpo import objective as objective_module
from hpo.objective import EvaluationConfig, TrialConfig, evaluate_greedy_q_net


class FakeTrial:
    number = 3

    def __init__(self) -> None:
        self.user_attrs = {}

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


class FakeEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEnvironmentFactory:
    def __init__(self) -> None:
        self.training_calls = []

    def make_training_env(self, num_envs):
        self.training_calls.append(num_envs)
        return FakeEnv()

    def evaluation_envs(self):
        return {"moon": lambda: FakeEnv(), "mars": lambda: FakeEnv()}


class FakeSearchSpace:
    def __init__(self) -> None:
        self.calls = []

    def training_config(
        self,
        trial,
        incumbent_params,
    ) -> VectorTrainingConfig:
        self.calls.append(("training_config", trial, incumbent_params))
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

    def replay_memory_capacity(self, trial, incumbent_params) -> int:
        self.calls.append(("replay_memory_capacity", trial, incumbent_params))
        return 12_345


@dataclass
class TrainerCall:
    env: FakeEnv
    seed: int | None
    device: object
    replay_memory_capacity: int
    training_config: object | None = None


def test_objective_trains_and_averages_named_evaluations(monkeypatch) -> None:
    calls = []
    eval_calls = []
    search_space = FakeSearchSpace()
    environment_factory = FakeEnvironmentFactory()

    class FakeTrainer:
        def __init__(
            self,
            env,
            *,
            seed,
            device,
            replay_memory_capacity,
        ) -> None:
            self.device = "trainer-device"
            calls.append(TrainerCall(env, seed, device, replay_memory_capacity))

        def train(self, training_config):
            calls[-1].training_config = training_config
            return VectorTrainingResult(
                q_net="fake-q-net",
                episode_returns=[10.0, 50.0],
                episode_lengths=[1, 1],
                episode_epsilons=[0.7, 0.6],
                env_steps=80,
                optimizer_updates=2,
            )

    scores = iter([120.0, 126.0])

    def score_fn(**kwargs):
        eval_calls.append(kwargs)
        return next(scores)

    monkeypatch.setattr(objective_module, "VectorTrainer", FakeTrainer)
    monkeypatch.setattr(objective_module, "evaluate_greedy_q_net", score_fn)

    objective = objective_module.create_objective(
        search_space=search_space,
        incumbent_params={"learning_rate": 0.001},
        environment_factory=environment_factory,
        trial_cfg=TrialConfig(num_envs=20, seed=100),
        evaluation_cfg=EvaluationConfig(),
    )

    trial = FakeTrial()
    score = objective(trial)

    assert search_space.calls == [
        ("training_config", trial, {"learning_rate": 0.001}),
        ("replay_memory_capacity", trial, {"learning_rate": 0.001}),
    ]
    assert score == pytest.approx(123.0)
    assert trial.user_attrs["world_scores"] == {"moon": 120.0, "mars": 126.0}
    assert trial.user_attrs["env_steps"] == 80
    assert trial.user_attrs["trial_seed"] == 103
    assert environment_factory.training_calls == [20]
    assert calls[0].env.closed
    assert calls[0].training_config.num_episodes == 12
    assert len(eval_calls) == 2
    assert all(call["episodes"] == 20 for call in eval_calls)


def test_single_evaluation_keeps_existing_trial_attributes(monkeypatch) -> None:
    class SingleEnvironmentFactory(FakeEnvironmentFactory):
        def evaluation_envs(self):
            return {"lunar_lander": lambda: FakeEnv()}

    class FakeTrainer:
        device = "cpu"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def train(self, _training_config):
            return VectorTrainingResult(
                q_net="fake-q-net",
                episode_returns=[1.0],
                episode_lengths=[1],
                episode_epsilons=[0.1],
                env_steps=1,
                optimizer_updates=1,
            )

    monkeypatch.setattr(objective_module, "VectorTrainer", FakeTrainer)
    monkeypatch.setattr(
        objective_module,
        "evaluate_greedy_q_net",
        lambda **_kwargs: 5.0,
    )

    objective = objective_module.create_objective(
        search_space=FakeSearchSpace(),
        incumbent_params={},
        environment_factory=SingleEnvironmentFactory(),
        trial_cfg=TrialConfig(seed=None),
        evaluation_cfg=EvaluationConfig(eval_episodes=7, eval_seed=50),
    )
    trial = FakeTrial()
    objective(trial)

    assert "world_scores" not in trial.user_attrs


def test_evaluate_greedy_q_net_returns_mean_episode_return() -> None:
    class FakeQNet:
        def eval(self) -> None:
            pass

        def __call__(self, _state):
            return torch.tensor([[0.0, 1.0]])

    class FakeEvalEnv:
        def __init__(self) -> None:
            self.closed = False

        def reset(self, *, seed=None):
            assert seed in {10, 11, 12}
            return [0.0], {}

        def step(self, action):
            assert action == 1
            return [0.0], 2.0, True, False, {}

        def close(self) -> None:
            self.closed = True

    envs = []

    def make_env():
        env = FakeEvalEnv()
        envs.append(env)
        return env

    score = evaluate_greedy_q_net(
        q_net=FakeQNet(),
        device=torch.device("cpu"),
        make_env=make_env,
        episodes=3,
        seed=10,
    )

    assert score == pytest.approx(2.0)
    assert all(env.closed for env in envs)
