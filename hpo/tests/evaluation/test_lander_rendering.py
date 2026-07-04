import gymnasium as gym

from hpo.evaluation.lander_rendering import LanderColors, LanderRenderWrapper


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
