import numpy as np
import pytest

from hpo.lunar_lander.environment import EnvFactory as LunarLanderEnvFactory
from hpo.solar_system_lander.environment import EnvFactory as SolarSystemLanderEnvFactory


def test_lunar_lander_factory_keeps_original_observation() -> None:
    env = LunarLanderEnvFactory().make_training_env(2)
    try:
        observations, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observations.shape == (2, 8)


def test_solar_system_lander_factory_balances_world_slots() -> None:
    env = SolarSystemLanderEnvFactory("8d").make_training_env(20)
    try:
        names = [wrapped.world.name for wrapped in env.envs]
    finally:
        env.close()

    assert {name: names.count(name) for name in set(names)} == {
        "moon": 4,
        "mercury": 4,
        "mars": 4,
        "earth": 4,
        "venus": 4,
    }


def test_solar_system_lander_requires_balanced_slot_count() -> None:
    with pytest.raises(ValueError, match="divisible by 5"):
        SolarSystemLanderEnvFactory("8d").make_training_env(16)


def test_solar_system_lander_11d_exposes_reproducible_weather() -> None:
    make_mars = SolarSystemLanderEnvFactory(
        "11d"
    ).evaluation_envs()["mars"]
    first = make_mars()
    second = make_mars()
    try:
        first_observation, _ = first.reset(seed=123)
        second_observation, _ = second.reset(seed=123)
    finally:
        first.close()
        second.close()

    assert first_observation.shape == (11,)
    assert np.array_equal(first_observation, second_observation)
    assert first_observation[-3] == pytest.approx(-3.8 / 12)
    assert 0 <= first_observation[-2] <= 4 / 20
    assert 0 <= first_observation[-1] <= 1 / 2


def test_solar_system_lander_8d_hides_world_parameters() -> None:
    env = SolarSystemLanderEnvFactory("8d").evaluation_envs()["earth"]()
    try:
        observation, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observation.shape == (8,)
