import numpy as np
import pytest

from hpo.solar_system_lander.environment import (
    DEFAULT_WORLD_MIX,
    EnvFactory,
    EnvWrapper,
    World,
    acceleration_vector,
)
from hpo.solar_system_lander.reward_shaping import GroundThrustPenaltyEnv


def test_solar_system_lander_factory_balances_world_slots() -> None:
    env = EnvFactory("8d", world_mix=DEFAULT_WORLD_MIX).make_training_env(20, params={})
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
        EnvFactory("8d", world_mix=DEFAULT_WORLD_MIX).make_training_env(16, params={})


def test_solar_system_lander_factory_accepts_weighted_world_mix() -> None:
    factory = EnvFactory("9d", world_mix={World.VENUS: 2, World.MOON: 1})
    env = factory.make_training_env(3, params={})
    try:
        names = [wrapped.world.name for wrapped in env.envs]
    finally:
        env.close()

    assert names == ["venus", "venus", "moon"]
    assert list(factory.evaluation_envs()) == ["venus", "moon"]
    assert factory.metadata()["worlds"][0]["wind_power"] == [15.0, 20.0]


def test_solar_system_lander_factory_wraps_training_envs_only() -> None:
    factory = EnvFactory(
        "8d",
        world_mix={World.MOON: 1},
        training_env_wrapper=lambda env, params: GroundThrustPenaltyEnv(
            env, ground_thrust_penalty=params["ground_thrust_penalty"]
        ),
    )
    training_env = factory.make_training_env(1, params={"ground_thrust_penalty": 0.5})
    evaluation_env = factory.evaluation_envs()["moon"]()
    try:
        assert isinstance(training_env.envs[0], GroundThrustPenaltyEnv)
        assert training_env.envs[0].ground_thrust_penalty == pytest.approx(0.5)
        assert isinstance(evaluation_env, EnvWrapper)
    finally:
        training_env.close()
        evaluation_env.close()


def test_acceleration_vector_uses_velocity_delta_and_clips() -> None:
    previous = np.array([0.0, 0.0, -0.7, 0.2], dtype=np.float32)
    current = np.array([0.0, 0.0, 0.6, -0.1], dtype=np.float32)

    acceleration = acceleration_vector(previous, current)

    assert acceleration.dtype == np.float32
    assert acceleration.tolist() == pytest.approx([1.0, -0.3])


def test_solar_system_lander_factory_rejects_empty_worlds() -> None:
    with pytest.raises(ValueError, match="world_mix must not be empty"):
        EnvFactory("8d", world_mix={})


def test_solar_system_lander_factory_rejects_zero_world_count() -> None:
    with pytest.raises(ValueError, match="world_mix count must be >= 1: moon"):
        EnvFactory("8d", world_mix={World.MOON: 0})


def test_solar_system_lander_11d_exposes_reproducible_weather() -> None:
    make_mars = EnvFactory("11d", world_mix=DEFAULT_WORLD_MIX).evaluation_envs()["mars"]
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
    env = EnvFactory("9d", world_mix=DEFAULT_WORLD_MIX).evaluation_envs()["mars"]()
    try:
        observation, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observation.shape == (9,)
    assert observation[-1] == pytest.approx(-3.8 / 12)


def test_solar_system_lander_10d_exposes_acceleration() -> None:
    env = EnvFactory("10d", world_mix=DEFAULT_WORLD_MIX).evaluation_envs()["earth"]()
    try:
        observation, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observation.shape == (10,)
    assert observation[-2:].tolist() == [0.0, 0.0]


def test_solar_system_lander_8d_hides_world_parameters() -> None:
    env = EnvFactory("8d", world_mix=DEFAULT_WORLD_MIX).evaluation_envs()["earth"]()
    try:
        observation, _ = env.reset(seed=42)
    finally:
        env.close()

    assert observation.shape == (8,)
