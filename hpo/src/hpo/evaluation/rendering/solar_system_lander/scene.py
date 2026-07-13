"""Custom rendering for Gymnasium LunarLander-compatible SolarSystemLander videos."""

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import lunar_lander
from gymnasium.error import DependencyNotInstalled

from hpo.evaluation.rendering.solar_system_lander.colors import LanderColors, LanderOverlay
from hpo.evaluation.rendering.solar_system_lander.env_state import EnvState
from hpo.evaluation.rendering.solar_system_lander.lander import LanderSkin, draw_gym_objects
from hpo.evaluation.rendering.solar_system_lander.overlay import draw_overlay
from hpo.evaluation.rendering.solar_system_lander.windsocks import draw_flags


class LanderRenderWrapper(gym.Wrapper):
    """Render a LunarLander-compatible environment with custom colors."""

    def __init__(
        self,
        env,
        colors: LanderColors | None = None,
        overlay: LanderOverlay | None = None,
        skin: LanderSkin | None = None,
    ) -> None:
        super().__init__(env)
        self.colors = colors or LanderColors()
        self.overlay = overlay
        self.skin = skin
        self.reset_seed: int | None = None
        self.steps_since_reset = 0
        self.score = 0.0

    def reset(self, *, seed: int | None = None, options=None):
        self.reset_seed = seed
        self.steps_since_reset = 0
        self.score = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.steps_since_reset += 1
        self.score += float(reward)
        return observation, reward, terminated, truncated, info

    def render(self):
        return _render_lunar_lander(self.env.unwrapped, self, self.colors, self.overlay, self.skin)


def _render_lunar_lander(
    env,
    source_env,
    colors: LanderColors,
    overlay: LanderOverlay | None,
    skin: LanderSkin | None = None,
):
    if env.render_mode is None:
        assert env.spec is not None
        gym.logger.warn(
            "You are calling render method without specifying any render mode. "
            "You can specify the render_mode at initialization, "
            f'e.g. gym.make("{env.spec.id}", render_mode="rgb_array")'
        )
        return None

    try:
        import pygame
        from pygame import gfxdraw
    except ImportError as error:
        raise DependencyNotInstalled(
            'pygame is not installed, run `pip install "gymnasium[box2d]"`'
        ) from error

    if env.screen is None and env.render_mode == "human":
        pygame.init()
        pygame.display.init()
        env.screen = pygame.display.set_mode((lunar_lander.VIEWPORT_W, lunar_lander.VIEWPORT_H))
    if env.clock is None:
        env.clock = pygame.time.Clock()

    env_state = EnvState.from_env(wrapper=source_env, env=env)
    env.surf = pygame.Surface((lunar_lander.VIEWPORT_W, lunar_lander.VIEWPORT_H))

    pygame.transform.scale(env.surf, (lunar_lander.SCALE, lunar_lander.SCALE))
    pygame.draw.rect(env.surf, colors.ground, env.surf.get_rect())

    _prepare_particles(env)
    env._clean_particles(False)
    _draw_sky(env, colors, gfxdraw)
    draw_gym_objects(env.surf, env, colors, gfxdraw, hide_lander_body=skin is not None)
    draw_flags(env.surf, env, colors, env_state.wind, gfxdraw)

    env.surf = pygame.transform.flip(env.surf, False, True)
    if skin is not None:
        skin.draw(env.surf, env)
    if overlay is not None:
        draw_overlay(env.surf, env_state, overlay)

    if env.render_mode == "human":
        assert env.screen is not None
        env.screen.blit(env.surf, (0, 0))
        pygame.event.pump()
        env.clock.tick(env.metadata["render_fps"])
        pygame.display.flip()
        return None
    if env.render_mode == "rgb_array":
        return np.transpose(np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2))

    return None


def _prepare_particles(env) -> None:
    for obj in env.particles:
        obj.ttl -= 0.15
        obj.color1 = (
            int(max(0.2, 0.15 + obj.ttl) * 255),
            int(max(0.2, 0.5 * obj.ttl) * 255),
            int(max(0.2, 0.5 * obj.ttl) * 255),
        )
        obj.color2 = (
            int(max(0.2, 0.15 + obj.ttl) * 255),
            int(max(0.2, 0.5 * obj.ttl) * 255),
            int(max(0.2, 0.5 * obj.ttl) * 255),
        )


def _draw_sky(env, colors: LanderColors, gfxdraw) -> None:
    import pygame

    for polygon in env.sky_polys:
        scaled_poly = []
        for coord in polygon:
            scaled_poly.append((coord[0] * lunar_lander.SCALE, coord[1] * lunar_lander.SCALE))
        pygame.draw.polygon(env.surf, colors.sky, scaled_poly)
        gfxdraw.aapolygon(env.surf, scaled_poly, colors.sky)
        pygame.draw.aaline(env.surf, colors.ground_outline, scaled_poly[0], scaled_poly[1])
