from hpo.environments.lunar_lander.env import EnvFactory


def test_lunar_lander_factory_keeps_original_observation() -> None:
    env = EnvFactory().make_training_env(2, params={})
    try:
        observations, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observations.shape == (2, 8)
