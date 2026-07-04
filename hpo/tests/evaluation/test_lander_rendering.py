import gymnasium as gym
import pytest

from hpo.evaluation.lander_rendering import (
    LanderColors,
    LanderRenderWrapper,
    _overlay_lines,
    world_colors,
)
from hpo.solar_system_lander.environment import World


def test_lander_render_wrapper_uses_custom_sky_and_ground_colors():
    colors = LanderColors(
        sky=(10, 20, 30),
        ground=(40, 50, 60),
        ground_outline=(40, 50, 60),
    )
    env = LanderRenderWrapper(
        gym.make("LunarLander-v3", render_mode="rgb_array"),
        colors,
    )

    try:
        env.reset(seed=123)
        frame = env.render()
    finally:
        env.close()

    assert frame.shape == (400, 600, 3)
    assert tuple(frame[10, 10]) == colors.sky
    assert tuple(frame[390, 10]) == colors.ground


def test_lander_render_wrapper_does_not_draw_vertical_sky_polygon_seams():
    colors = LanderColors(
        sky=(10, 20, 30),
        ground=(40, 50, 60),
        ground_outline=(40, 50, 60),
    )
    env = LanderRenderWrapper(
        gym.make("LunarLander-v3", render_mode="rgb_array"),
        colors,
    )

    try:
        env.reset(seed=123)
        frame = env.render()
    finally:
        env.close()

    assert tuple(frame[10, 60]) == colors.sky


def test_overlay_lines_include_static_world_conditions():
    class World:
        name = "earth"
        gravity = -10.0

    class Lander:
        mass = 4.8
        inertia = 0.8

    class Unwrapped:
        lander = Lander()

    class Env:
        world = World()
        _weather = (12.36, 1.14)
        unwrapped = Unwrapped()
        reset_seed = 123

    assert _overlay_lines(Env()) == [
        "Earth",
        "g: 10.0 m/s²",
        "wind: 2.6 m/s²",
        "turb: 1.4 rad/s²",
        "kick: 3.6 m/s",
    ]


def test_world_colors_returns_colors_in_world_order():
    colors = world_colors([World.EARTH, World.VENUS])

    assert colors == [
        LanderColors(
            sky=(143, 199, 232),
            ground=(111, 127, 82),
            ground_outline=(111, 127, 82),
        ),
        LanderColors(
            sky=(214, 168, 92),
            ground=(138, 103, 64),
            ground_outline=(138, 103, 64),
        ),
    ]


def test_world_colors_rejects_unknown_world():
    with pytest.raises(ValueError, match="unknown world color: pluto"):
        world_colors(["pluto"])
