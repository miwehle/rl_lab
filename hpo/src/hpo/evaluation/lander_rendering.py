"""Custom rendering colors for Gymnasium LunarLander videos."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import lunar_lander
from gymnasium.error import DependencyNotInstalled

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class LanderColors:
    sky: RGB = (0, 0, 0)
    ground: RGB = (255, 255, 255)
    ground_outline: RGB = (0, 0, 0)
    lander_fill: RGB = (128, 102, 230)
    lander_outline: RGB = (77, 77, 128)
    flag_pole: RGB = (255, 255, 255)
    flag: RGB = (204, 204, 0)


class LanderRenderWrapper(gym.Wrapper):
    """Render a LunarLander-compatible environment with custom colors."""

    def __init__(self, env, colors: LanderColors) -> None:
        super().__init__(env)
        self.colors = colors

    def render(self):
        return _render_lunar_lander(self.env.unwrapped, self.colors)


def _render_lunar_lander(env, colors: LanderColors):
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
        env.screen = pygame.display.set_mode(
            (lunar_lander.VIEWPORT_W, lunar_lander.VIEWPORT_H)
        )
    if env.clock is None:
        env.clock = pygame.time.Clock()

    env.surf = pygame.Surface((lunar_lander.VIEWPORT_W, lunar_lander.VIEWPORT_H))

    pygame.transform.scale(env.surf, (lunar_lander.SCALE, lunar_lander.SCALE))
    pygame.draw.rect(env.surf, colors.ground, env.surf.get_rect())

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

    env._clean_particles(False)

    for p in env.sky_polys:
        scaled_poly = []
        for coord in p:
            scaled_poly.append(
                (coord[0] * lunar_lander.SCALE, coord[1] * lunar_lander.SCALE)
            )
        pygame.draw.polygon(env.surf, colors.sky, scaled_poly)
        gfxdraw.aapolygon(env.surf, scaled_poly, colors.sky)
        pygame.draw.aaline(
            env.surf,
            colors.ground_outline,
            scaled_poly[0],
            scaled_poly[1],
        )

    for obj in env.particles + env.drawlist:
        for f in obj.fixtures:
            trans = f.body.transform
            fill, outline = _object_colors(obj, env, colors)
            if type(f.shape) is lunar_lander.circleShape:
                pygame.draw.circle(
                    env.surf,
                    color=fill,
                    center=trans * f.shape.pos * lunar_lander.SCALE,
                    radius=f.shape.radius * lunar_lander.SCALE,
                )
                pygame.draw.circle(
                    env.surf,
                    color=outline,
                    center=trans * f.shape.pos * lunar_lander.SCALE,
                    radius=f.shape.radius * lunar_lander.SCALE,
                )
            else:
                path = [trans * v * lunar_lander.SCALE for v in f.shape.vertices]
                pygame.draw.polygon(env.surf, color=fill, points=path)
                gfxdraw.aapolygon(env.surf, path, fill)
                pygame.draw.aalines(
                    env.surf, color=outline, points=path, closed=True
                )

            for x in [env.helipad_x1, env.helipad_x2]:
                x = x * lunar_lander.SCALE
                flagy1 = env.helipad_y * lunar_lander.SCALE
                flagy2 = flagy1 + 50
                pygame.draw.line(
                    env.surf,
                    color=colors.flag_pole,
                    start_pos=(x, flagy1),
                    end_pos=(x, flagy2),
                    width=1,
                )
                flag_points = [
                    (x, flagy2),
                    (x, flagy2 - 10),
                    (x + 25, flagy2 - 5),
                ]
                pygame.draw.polygon(env.surf, color=colors.flag, points=flag_points)
                gfxdraw.aapolygon(env.surf, flag_points, colors.flag)

    env.surf = pygame.transform.flip(env.surf, False, True)

    if env.render_mode == "human":
        assert env.screen is not None
        env.screen.blit(env.surf, (0, 0))
        pygame.event.pump()
        env.clock.tick(env.metadata["render_fps"])
        pygame.display.flip()
        return None
    if env.render_mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2)
        )

    return None


def _object_colors(obj, env, colors: LanderColors) -> tuple[RGB, RGB]:
    if obj is env.lander or any(obj is leg for leg in env.legs):
        return colors.lander_fill, colors.lander_outline
    return obj.color1, obj.color2
