import gymnasium as gym
import pytest

from hpo.evaluation.lander_rendering import (
    LanderColors,
    LanderRenderWrapper,
    world_colors,
)
from hpo.solar_system_lander.environment import World


class ScoreEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,))

    def reset(self, *, seed=None, options=None):
        return [0.0], {}

    def step(self, action):
        return [0.0], 1.25, False, False, {}


def test_world_colors_returns_colors_in_world_order():
    colors = world_colors([World.EARTH, World.VENUS])

    assert colors == [
        LanderColors(sky=(143, 199, 232), ground=(111, 127, 82), ground_outline=(111, 127, 82)),
        LanderColors(sky=(214, 168, 92), ground=(138, 103, 64), ground_outline=(138, 103, 64)),
    ]


def test_world_colors_rejects_unknown_world():
    with pytest.raises(ValueError, match="unknown world color: pluto"):
        world_colors(["pluto"])


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
