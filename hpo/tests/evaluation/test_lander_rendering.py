import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.box2d import lunar_lander

from hpo.evaluation.lander_rendering import (
    LanderColors,
    LanderOverlay,
    LanderRenderWrapper,
    _force_color,
    _wind_strength,
    world_colors,
)
from hpo.evaluation.lander_skins import DetailedEagleSkin
from hpo.solar_system_lander.environment import EnvFactory, World


class ScoreEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    def reset(self, *, seed=None, options=None):
        return [0.0], {}

    def step(self, action):
        return [0.0], 1.25, False, False, {}


class MarkerSkin:
    def __init__(self):
        self.calls = []

    def draw(self, surface, env):
        self.calls.append(env)
        surface.set_at((7, 9), (255, 0, 0))


def test_world_colors_returns_colors_in_world_order():
    colors = world_colors([World.EARTH, World.VENUS])

    assert colors == [
        LanderColors(sky=(143, 199, 232), ground=(111, 127, 82), ground_outline=(111, 127, 82)),
        LanderColors(sky=(214, 168, 92), ground=(138, 103, 64), ground_outline=(138, 103, 64)),
    ]


def test_world_colors_rejects_unknown_world():
    with pytest.raises(ValueError, match="unknown world color: pluto"):
        world_colors(["pluto"])


def test_force_color_uses_absolute_stops():
    stops = ((0.0, (0, 255, 0)), (1.0, (255, 255, 0)), (2.0, (255, 0, 0)))

    assert _force_color(0.0, stops) == (0, 255, 0)
    assert _force_color(1.0, stops) == (255, 255, 0)
    assert _force_color(2.0, stops) == (255, 0, 0)
    assert _force_color(3.0, stops) == (255, 0, 0)


def test_wind_strength_follows_current_wind_sign():
    class Env:
        enable_wind = True
        wind_power = 1.0

    env = Env()
    env.wind_idx = 25
    positive, max_wind = _wind_strength(env)
    assert positive > 0
    assert max_wind == 1.0

    env.wind_idx = 150
    negative, max_wind = _wind_strength(env)
    assert negative < 0
    assert max_wind == 1.0


class TestLanderRenderWrapper:
    def test_uses_custom_sky_and_ground_colors(self):
        colors = LanderColors(sky=(10, 20, 30), ground=(40, 50, 60), ground_outline=(40, 50, 60))
        env = LanderRenderWrapper(gym.make("LunarLander-v3", render_mode="rgb_array"), colors)

        try:
            env.reset(seed=123)
            frame = env.render()
        finally:
            env.close()

        assert frame.shape == (400, 600, 3)
        assert tuple(frame[10, 10]) == colors.sky
        assert tuple(frame[390, 10]) == colors.ground

    def test_does_not_draw_vertical_sky_polygon_seams(self):
        colors = LanderColors(sky=(10, 20, 30), ground=(40, 50, 60), ground_outline=(40, 50, 60))
        env = LanderRenderWrapper(gym.make("LunarLander-v3", render_mode="rgb_array"), colors)

        try:
            env.reset(seed=123)
            frame = env.render()
        finally:
            env.close()

        assert tuple(frame[10, 60]) == colors.sky

    def test_accumulates_score_and_resets_it(self):
        env = LanderRenderWrapper(ScoreEnv())

        env.reset(seed=123)
        env.step(0)
        env.step(0)

        assert env.score == 2.5

        env.reset(seed=123)

        assert env.score == 0.0

    def test_overlay_draws_wind_and_turbulence_indicators(self):
        plain_frame, _ = _render_venus_frame(overlay=None)
        overlay_frame, lander_center = _render_venus_frame(overlay=LanderOverlay())

        top_center = np.s_[8:52, 200:400]
        assert np.any(overlay_frame[top_center] != plain_frame[top_center])

        x, y = lander_center
        lander_neighborhood = np.s_[max(y - 70, 0) : y + 30, x : min(x + 80, overlay_frame.shape[1])]
        assert np.any(overlay_frame[lander_neighborhood] != plain_frame[lander_neighborhood])

    def test_draws_optional_skin_on_screen_oriented_surface(self):
        skin = MarkerSkin()
        env = LanderRenderWrapper(gym.make("LunarLander-v3", render_mode="rgb_array"), skin=skin)

        try:
            env.reset(seed=123)
            frame = env.render()
        finally:
            env.close()

        assert len(skin.calls) == 1
        assert tuple(frame[9, 7]) == (255, 0, 0)

    def test_detailed_eagle_skin_renders_frame(self):
        env = LanderRenderWrapper(
            gym.make("LunarLander-v3", render_mode="rgb_array"),
            colors=world_colors(["earth"])[0],
            skin=DetailedEagleSkin(),
        )

        try:
            env.reset(seed=10_014)
            frame = env.render()
        finally:
            env.close()

        assert frame.shape == (400, 600, 3)
        assert np.any(frame != world_colors(["earth"])[0].sky)

    def test_detailed_eagle_skin_auto_halo_is_only_for_dark_backgrounds(self):
        moon_auto = _render_skin_frame(World.MOON, DetailedEagleSkin(halo="auto"))
        moon_never = _render_skin_frame(World.MOON, DetailedEagleSkin(halo="never"))
        earth_auto = _render_skin_frame(World.EARTH, DetailedEagleSkin(halo="auto"))
        earth_never = _render_skin_frame(World.EARTH, DetailedEagleSkin(halo="never"))

        assert _frame_delta(moon_auto, moon_never) > 0
        assert _frame_delta(earth_auto, earth_never) == 0

    def test_detailed_eagle_skin_rejects_unknown_halo_mode(self):
        with pytest.raises(ValueError, match="unknown halo mode"):
            DetailedEagleSkin(halo="full neon")  # type: ignore[arg-type]


def _render_venus_frame(*, overlay):
    factory = EnvFactory("10d", world_mix={World.VENUS: 1})
    colors = world_colors([World.VENUS])[0]
    env = LanderRenderWrapper(
        factory.make_env(World.VENUS, render_mode="rgb_array"),
        colors=colors,
        overlay=overlay,
    )
    try:
        env.reset(seed=10_173)
        env.step(0)
        lander = env.env.unwrapped.lander
        lander_center = (
            round(lander.position.x * lunar_lander.SCALE),
            round(lunar_lander.VIEWPORT_H - lander.position.y * lunar_lander.SCALE),
        )
        return env.render(), lander_center
    finally:
        env.close()


def _render_skin_frame(world: World, skin: DetailedEagleSkin):
    factory = EnvFactory("10d", world_mix={world: 1})
    env = LanderRenderWrapper(
        factory.make_env(world, render_mode="rgb_array"),
        colors=world_colors([world])[0],
        skin=skin,
    )
    try:
        env.reset(seed=10_014)
        for _ in range(45):
            env.step(0)
        return env.render()
    finally:
        env.close()


def _frame_delta(left, right) -> int:
    return int(np.abs(left.astype(np.int16) - right.astype(np.int16)).sum())
