import numpy as np
import pytest

from hpo.solar_system_lander.environment import (
    EnvFactory,
    World,
    WorldConfig,
    acceleration_vector,
    worlds_by_name,
)


def test_solar_system_lander_factory_balances_world_slots() -> None:
    env = EnvFactory("8d").make_training_env(20)
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
        EnvFactory("8d").make_training_env(16)


def test_solar_system_lander_factory_accepts_custom_worlds() -> None:
    calm_venus = WorldConfig("calm_venus", -9.0, (0.0, 0.0), (0.0, 0.0))
    factory = EnvFactory("9d", worlds=(calm_venus,))
    env = factory.make_training_env(3)
    try:
        names = [wrapped.world.name for wrapped in env.envs]
    finally:
        env.close()

    assert names == ["calm_venus", "calm_venus", "calm_venus"]
    assert list(factory.evaluation_envs()) == ["calm_venus"]
    assert factory.metadata()["worlds"][0]["wind_power"] == [0.0, 0.0]


def test_worlds_by_name_returns_requested_worlds() -> None:
    assert [world.name for world in worlds_by_name(World.EARTH, World.VENUS)] == [
        World.EARTH,
        World.VENUS,
    ]


def test_worlds_by_name_rejects_unknown_world() -> None:
    with pytest.raises(ValueError, match="unknown world: pluto"):
        worlds_by_name("pluto")


def test_acceleration_vector_uses_velocity_delta_and_clips() -> None:
    previous = np.array([0.0, 0.0, -0.7, 0.2], dtype=np.float32)
    current = np.array([0.0, 0.0, 0.6, -0.1], dtype=np.float32)

    acceleration = acceleration_vector(previous, current)

    assert acceleration.dtype == np.float32
    assert acceleration.tolist() == pytest.approx([1.0, -0.3])


def test_solar_system_lander_factory_rejects_empty_worlds() -> None:
    with pytest.raises(ValueError, match="worlds must not be empty"):
        EnvFactory("8d", worlds=())


def test_solar_system_lander_11d_exposes_reproducible_weather() -> None:
    make_mars = EnvFactory("11d").evaluation_envs()["mars"]
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


def test_solar_system_lander_9d_exposes_gravity() -> None:
    env = EnvFactory("9d").evaluation_envs()["mars"]()
    try:
        observation, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observation.shape == (9,)
    assert observation[-1] == pytest.approx(-3.8 / 12)


def test_solar_system_lander_10d_exposes_acceleration() -> None:
    env = EnvFactory("10d").evaluation_envs()["earth"]()
    try:
        observation, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observation.shape == (10,)
    assert observation[-2:].tolist() == [0.0, 0.0]


def test_solar_system_lander_8d_hides_world_parameters() -> None:
    env = EnvFactory("8d").evaluation_envs()["earth"]()
    try:
        observation, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observation.shape == (8,)
