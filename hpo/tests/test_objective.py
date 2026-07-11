import json
from types import SimpleNamespace

import pytest
import torch

from dqn.vector_training import VectorTrainingConfig, VectorTrainingResult
from hpo import objective as objective_module
from hpo.objective import evaluate_greedy_q_net
from hpo.study_reporting import TrainingProgressPlotter
from common import objective_config


class FakeTrial:
    number = 3

    def __init__(self) -> None:
        self.params = {}
        self.user_attrs = {}

    def suggest_float(self, name, low, high, *, log=False):
        self.params[name] = low
        return low

    def suggest_int(self, name, low, high, *, log=False):
        self.params[name] = low
        return low

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


BASELINE_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "eps_start": 0.7,
    "eps_end": 0.02,
    "eps_decay": 1234,
    "gamma": 0.99,
    "tau": 0.005,
    "learning_starts": 77,
    "optimize_every": 3,
    "replay_memory": 12_345,
    "num_episodes": 12,
}


def vector_result(
    episode_returns=None,
    *,
    q_net="fake-q-net",
    episode_epsilons=None,
    episode_env_indices=None,
    env_steps=None,
    optimizer_updates=1,
    early_stopped=False,
    early_stopping_score=None,
) -> VectorTrainingResult:
    returns = episode_returns or [1.0]
    episodes = len(returns)
    return VectorTrainingResult(
        q_net=q_net,
        episode_returns=returns,
        episode_lengths=[1] * episodes,
        episode_epsilons=episode_epsilons or [0.1] * episodes,
        episode_env_indices=episode_env_indices or [0] * episodes,
        env_steps=env_steps or episodes,
        optimizer_updates=optimizer_updates,
        early_stopped=early_stopped,
        early_stopping_score=early_stopping_score,
    )


def create_test_objective(*, suggest_parameter_values=None, environment_factory=None, **config_overrides):
    return objective_module.create_objective(
        suggest_parameter_values=suggest_parameter_values or FakeSuggestParameterValues(),
        incumbent_params=BASELINE_PARAMS,
        config=objective_config(
            environment_factory=environment_factory or FakeEnvironmentFactory(), **config_overrides
        ),
    )


def trainer_class(result, *, on_train=None):
    class FakeTrainer:
        device = "cpu"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def train(self, training_config, *, plotter=None):
            assert plotter is None
            if on_train is not None:
                on_train(training_config)
            return result

    return FakeTrainer


class StaticHookFactory:
    def __init__(self, hooks, on_trial=lambda _ctx: None) -> None:
        self.hooks = hooks
        self.on_trial = on_trial

    def for_trial(self, ctx):
        self.on_trial(ctx)
        return self.hooks

    def study_attrs(self):
        return {}


def test_objective_config_study_attrs_are_json_serializable() -> None:
    attrs = objective_config(device=torch.device("cuda")).study_attrs()

    assert attrs["device"] == "cuda"
    json.dumps(attrs)


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


class FakeSuggestParameterValues:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, trial, incumbent_params) -> None:
        self.calls.append((trial, incumbent_params))
        trial.suggest_float("learning_rate", 5e-4, 1e-3)


def test_objective_trains_and_averages_named_evaluations(monkeypatch) -> None:
    calls = []
    eval_calls = []
    suggest_parameter_values = FakeSuggestParameterValues()
    environment_factory = FakeEnvironmentFactory()

    class FakeTrainer:
        def __init__(self, env, *, seed, device, replay_memory_capacity, hidden_size) -> None:
            self.device = "trainer-device"
            calls.append(SimpleNamespace(env=env))

        def train(self, training_config, *, plotter=None):
            assert plotter is None
            calls[-1].training_config = training_config
            return vector_result(
                [10.0, 50.0],
                episode_epsilons=[0.7, 0.6],
                episode_env_indices=[0, 1],
                env_steps=80,
                optimizer_updates=2,
            )

    scores = iter([120.0, 126.0])

    def score_fn(**kwargs):
        eval_calls.append(kwargs)
        return next(scores)

    monkeypatch.setattr(objective_module, "VectorTrainer", FakeTrainer)
    monkeypatch.setattr(objective_module, "evaluate_greedy_q_net", score_fn)

    objective = create_test_objective(
        suggest_parameter_values=suggest_parameter_values,
        environment_factory=environment_factory,
        num_envs=20,
        training_seed=100,
    )

    trial = FakeTrial()
    score = objective(trial)

    assert suggest_parameter_values.calls == [(trial, BASELINE_PARAMS)]
    assert score == pytest.approx(123.0)
    assert trial.user_attrs["world_scores"] == {"moon": 120.0, "mars": 126.0}
    assert trial.user_attrs["env_steps"] == 80
    assert trial.user_attrs["trained_episodes"] == 2
    assert trial.user_attrs["trial_seed"] == 103
    assert environment_factory.training_calls == [20]
    assert calls[0].env.closed
    assert calls[0].training_config.num_episodes == 12
    assert calls[0].training_config.adaptive_extension_window == 50
    assert calls[0].training_config.early_stopping_score == pytest.approx(-250.0)
    assert calls[0].training_config.hidden_size == 128
    assert calls[0].training_config.learning_rate == 5e-4
    assert calls[0].training_config.eps_start == pytest.approx(0.7)
    assert len(eval_calls) == 2
    assert all(call["episodes"] == 20 for call in eval_calls)


def test_objective_returns_early_stopping_score_without_evaluation(monkeypatch) -> None:
    def assert_early_stopping_config(training_config):
        assert training_config.early_stopping_score == pytest.approx(-250.0)

    monkeypatch.setattr(
        objective_module,
        "VectorTrainer",
        trainer_class(
            vector_result([-300.0] * 6, early_stopped=True, early_stopping_score=-300.0),
            on_train=assert_early_stopping_config,
        ),
    )
    monkeypatch.setattr(
        objective_module,
        "evaluate_greedy_q_net",
        lambda **_kwargs: pytest.fail("early-stopped trials are not evaluated"),
    )

    objective = create_test_objective()
    trial = FakeTrial()

    assert objective(trial) == pytest.approx(-300.0)
    assert trial.user_attrs["early_stopped"] is True
    assert trial.user_attrs["early_stopping_score"] == pytest.approx(-300.0)
    assert trial.user_attrs["trained_episodes"] == 6
    assert "world_scores" not in trial.user_attrs


def test_single_evaluation_keeps_existing_trial_attributes(monkeypatch) -> None:
    class SingleEnvironmentFactory(FakeEnvironmentFactory):
        def evaluation_envs(self):
            return {"lunar_lander": lambda: FakeEnv()}

    monkeypatch.setattr(objective_module, "VectorTrainer", trainer_class(vector_result()))
    monkeypatch.setattr(objective_module, "evaluate_greedy_q_net", lambda **_kwargs: 5.0)

    objective = create_test_objective(
        environment_factory=SingleEnvironmentFactory(), training_seed=None, eval_episodes=7, eval_seed=50
    )
    trial = FakeTrial()
    objective(trial)

    assert "world_scores" not in trial.user_attrs


def test_objective_uses_objective_hooks(monkeypatch) -> None:
    suggest = FakeSuggestParameterValues()
    hook_calls = []

    class FakeQNet:
        hooked = False

    class FakeTrainer:
        device = "cpu"

        def __init__(self, _env, *, seed, device, replay_memory_capacity, hidden_size) -> None:
            hook_calls.append((seed, device, replay_memory_capacity, hidden_size))

        def train(self, _training_config, *, plotter=None):
            assert plotter is None
            return vector_result(q_net=FakeQNet())

    class FakeHooks:
        def make_trainer(self, *args, **kwargs):
            return FakeTrainer(*args, **kwargs)

        def q_net_for_evaluation(self, ctx):
            ctx.training_result.q_net.hooked = True
            return ctx.training_result.q_net

        def training_plotter(self):
            return None

        def finalize_trial(self, ctx):
            assert ctx.score == pytest.approx(10.0)
            assert ctx.world_scores == {"moon": 10.0, "mars": 10.0}
            assert ctx.training_result is not None
            ctx.trial.set_user_attr("hook_attr", "yes")

    def score_fn(**kwargs):
        assert kwargs["q_net"].hooked
        return 10.0

    monkeypatch.setattr(objective_module, "evaluate_greedy_q_net", score_fn)

    hook_factory = StaticHookFactory(FakeHooks(), lambda c: hook_calls.append((c.trial, c.training_config)))
    objective = create_test_objective(suggest_parameter_values=suggest, hooks=hook_factory, eval_episodes=1)
    trial = FakeTrial()

    assert objective(trial) == pytest.approx(10.0)
    assert hook_calls[0][0] is trial
    assert isinstance(hook_calls[0][1], VectorTrainingConfig)
    assert hook_calls[1] == (45, None, 12_345, 128)
    assert trial.user_attrs["hook_attr"] == "yes"


def test_objective_reports_live_training_progress(monkeypatch) -> None:
    progress_calls = []

    class FakeTrainer:
        device = "cpu"

        def __init__(self, hooks) -> None:
            self.hooks = hooks

        def train(self, _training_config, *, plotter):
            plotter.plot_returns([1.0], epsilons=[0.9], env_indices=[0])
            self.hooks.best_checkpoint_score = 4.0
            plotter.plot_returns([1.0, 5.0], epsilons=[0.9, 0.8], env_indices=[0, 1])
            return vector_result([1.0, 5.0], episode_env_indices=[0, 1])

    class FakeHooks:
        best_checkpoint_score = None

        def make_trainer(self, *_args, **_kwargs):
            return FakeTrainer(self)

        def q_net_for_evaluation(self, ctx):
            return ctx.training_result.q_net

        def training_plotter(self):
            return TrainingProgressPlotter(
                trial_number=3,
                target_episodes=12,
                progress_fn=progress_calls.append,
                checkpoint_window=2,
                checkpoint_min_score=3.0,
                best_checkpoint_score=lambda: self.best_checkpoint_score,
                env_labels=["moon", "mars"],
            )

        def finalize_trial(self, _ctx):
            pass

    monkeypatch.setattr(objective_module, "evaluate_greedy_q_net", lambda **_kwargs: 10.0)

    objective = create_test_objective(hooks=StaticHookFactory(FakeHooks()), eval_episodes=1)

    objective(FakeTrial())

    assert [progress.episode_returns for progress in progress_calls] == [[1.0], [1.0, 5.0]]
    assert [progress.episode_epsilons for progress in progress_calls] == [[0.9], [0.9, 0.8]]
    assert [progress.episode_env_labels for progress in progress_calls] == [["moon"], ["moon", "mars"]]
    assert progress_calls[0].checkpoint_window == 2
    assert progress_calls[0].checkpoint_min_score == pytest.approx(3.0)
    assert progress_calls[0].best_checkpoint_score is None
    assert progress_calls[1].best_checkpoint_score == pytest.approx(4.0)


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
        q_net=FakeQNet(), device=torch.device("cpu"), make_env=make_env, episodes=3, seed=10
    )

    assert score == pytest.approx(2.0)
    assert all(env.closed for env in envs)
