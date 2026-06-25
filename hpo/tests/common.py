from hpo.objective import ObjectiveConfig


class DummyEnvironmentFactory:
    pass


def objective_config(**overrides) -> ObjectiveConfig:
    defaults = {
        "environment_factory": DummyEnvironmentFactory(),
        "num_envs": 16,
        "eval_episodes": 20,
    }
    return ObjectiveConfig(**(defaults | overrides))
