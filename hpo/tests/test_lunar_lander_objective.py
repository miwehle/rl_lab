from dataclasses import dataclass

import pytest
import torch

from dqn.vector_training import VectorTrainingConfig, VectorTrainingResult
from hpo.evaluation.scoring import ScoringConfig
from hpo.lunar_lander import objective as objective_module
from hpo.lunar_lander.objective import TrialConfig, evaluate_greedy_policy


class FakeTrial:
    number = 3

    def __init__(self) -> None:
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, *, log=False):
        return low

    def suggest_int(self, name, low, high, *, log=False):
        return low

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


class FakeEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeSearchSpace:
    def __init__(self) -> None:
        self.calls = []

    def training_config(self, trial, num_episodes: int) -> VectorTrainingConfig:
        self.calls.append(("training_config", trial, num_episodes))
        return VectorTrainingConfig(
            num_episodes=num_episodes,
            batch_size=64,
            eps_start=0.7,
            eps_end=0.02,
            eps_decay=1234,
            learning_rate=5e-4,
            learning_starts=77,
            optimize_every=3,
        )

    def replay_memory_capacity(self, trial) -> int:
        self.calls.append(("replay_memory_capacity", trial))
        return 12_345


@dataclass
class TrainerCall:
    env: FakeEnv
    seed: int | None
    device: object
    replay_memory_capacity: int
    training_config: object | None = None


def test_lunar_lander_objective_trains_vector_trial_and_returns_score(monkeypatch) -> None:
    calls = []
    envs = []
    eval_calls = []
    search_space = FakeSearchSpace()

    def vector_env_factory(env_id, num_envs):
        assert env_id == "LunarLander-v3"
        assert num_envs == 16
        env = FakeEnv()
        envs.append(env)
        return env

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
            calls.append(
                TrainerCall(
                    env,
                    seed,
                    device,
                    replay_memory_capacity,
                )
            )

        def train(self, training_config):
            calls[-1].training_config = training_config
            return VectorTrainingResult(
                q_net="fake-q-net",
                episode_returns=[10.0, 50.0, 40.0, 20.0],
                episode_lengths=[1, 1, 1, 1],
                episode_epsilons=[0.7, 0.6, 0.5, 0.4],
                env_steps=80,
                optimizer_updates=2,
            )

    def gym_score_fn(**kwargs):
        eval_calls.append(kwargs)
        return 123.0

    monkeypatch.setattr(objective_module, "_make_vector_env", vector_env_factory)
    monkeypatch.setattr(objective_module, "VectorTrainer", FakeTrainer)
    monkeypatch.setattr(objective_module, "evaluate_greedy_policy", gym_score_fn)

    objective = objective_module.create_objective(
        search_space=search_space,
        trial_cfg=TrialConfig(num_episodes=12, seed=100),
        scoring_cfg=ScoringConfig(
            baseline_env_steps=100,
            baseline_processed_samples=100,
        ),
    )

    trial = FakeTrial()
    score = objective(trial)

    assert search_space.calls == [
        ("training_config", trial, 12),
        ("replay_memory_capacity", trial),
    ]
    assert score == pytest.approx(-1.39)
    assert trial.user_attrs["gym_score"] == pytest.approx(123.0)
    assert trial.user_attrs["env_steps"] == 80
    assert trial.user_attrs["optimizer_updates"] == 2
    assert trial.user_attrs["processed_samples"] == 128
    assert trial.user_attrs["training_effort"] == pytest.approx(1.04)
    assert "objective_score" not in trial.user_attrs
    assert trial.user_attrs["trial_seed"] == 103
    assert trial.user_attrs["wall_time_seconds"] >= 0.0
    assert trial.user_attrs["training_curve"] == {
        "episode_returns": [10.0, 50.0, 40.0, 20.0],
        "episode_epsilons": [0.7, 0.6, 0.5, 0.4],
    }
    assert envs[0].closed
    assert calls[0].seed == 103
    assert calls[0].replay_memory_capacity == 12_345
    assert calls[0].training_config.num_episodes == 12
    assert calls[0].training_config.learning_rate == pytest.approx(5e-4)
    assert calls[0].training_config.learning_starts == 77
    assert calls[0].training_config.optimize_every == 3
    assert eval_calls[0]["q_net"] == "fake-q-net"
    assert eval_calls[0]["device"] == "trainer-device"
    assert eval_calls[0]["env_id"] == "LunarLander-v3"
    assert eval_calls[0]["episodes"] == 20
    assert eval_calls[0]["max_steps"] == 2_000
    assert eval_calls[0]["seed"] == 10_000


def test_lunar_lander_objective_passes_eval_settings_to_gym_score_fn(
    monkeypatch,
) -> None:
    eval_calls = []

    class FakeTrainer:
        device = "trainer-device"

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

    monkeypatch.setattr(
        objective_module,
        "_make_vector_env",
        lambda _env_id, _num_envs: FakeEnv(),
    )
    monkeypatch.setattr(objective_module, "VectorTrainer", FakeTrainer)
    monkeypatch.setattr(
        objective_module,
        "evaluate_greedy_policy",
        lambda **kwargs: eval_calls.append(kwargs) or 5.0,
    )

    objective = objective_module.create_objective(
        search_space=FakeSearchSpace(),
        trial_cfg=TrialConfig(num_episodes=1, seed=None),
        scoring_cfg=ScoringConfig(eval_episodes=7, eval_seed=50),
        eval_max_steps=99,
    )

    trial = FakeTrial()
    objective(trial)

    assert eval_calls[0]["episodes"] == 7
    assert eval_calls[0]["max_steps"] == 99
    assert eval_calls[0]["seed"] == 50
    assert trial.user_attrs["training_effort"] == 1.0


def test_evaluate_greedy_policy_returns_mean_episode_return() -> None:
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

    def env_factory(env_id):
        assert env_id == "LunarLander-v3"
        env = FakeEvalEnv()
        envs.append(env)
        return env

    score = evaluate_greedy_policy(
        q_net=FakeQNet(),
        device=torch.device("cpu"),
        env_id="LunarLander-v3",
        env_factory=env_factory,
        episodes=3,
        seed=10,
    )

    assert score == pytest.approx(2.0)
    assert all(env.closed for env in envs)
