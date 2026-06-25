from hpo.lunar_lander.environment import EnvFactory


def test_lunar_lander_factory_keeps_original_observation() -> None:
    env = EnvFactory().make_training_env(2)
    try:
        observations, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observations.shape == (2, 8)
