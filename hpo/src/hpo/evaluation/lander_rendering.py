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


@dataclass(frozen=True)
class LanderOverlay:
    text_color: RGB = (255, 255, 255)
    shadow_color: RGB = (0, 0, 0)


class LanderRenderWrapper(gym.Wrapper):
    """Render a LunarLander-compatible environment with custom colors."""

    def __init__(
        self,
        env,
        colors: LanderColors | None = None,
        overlay: LanderOverlay | None = None,
    ) -> None:
        super().__init__(env)
        self.colors = colors or LanderColors()
        self.overlay = overlay

    def render(self):
        return _render_lunar_lander(
            self.env.unwrapped,
            self.env,
            self.colors,
            self.overlay,
        )


def _render_lunar_lander(
    env,
    source_env,
    colors: LanderColors,
    overlay: LanderOverlay | None,
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
    if overlay is not None:
        _draw_overlay(env.surf, source_env, overlay)

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


def _draw_overlay(surface, source_env, overlay: LanderOverlay) -> None:
    import pygame

    lines = _overlay_lines(source_env)
    if not lines:
        return

    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, 18)
    x, y = 8, 8
    line_height = font.get_linesize()
    for index, line in enumerate(lines):
        position = (x, y + index * line_height)
        shadow = font.render(line, True, overlay.shadow_color)
        text = font.render(line, True, overlay.text_color)
        surface.blit(shadow, (position[0] + 1, position[1] + 1))
        surface.blit(text, position)


def _overlay_lines(source_env) -> list[str]:
    lines = []
    world = getattr(source_env, "world", None)
    if world is not None:
        name = getattr(world, "name", None)
        gravity = getattr(world, "gravity", None)
        if name is not None:
            lines.append(str(name).title())
        if gravity is not None:
            lines.append(f"g: {abs(float(gravity)):.1f} m/s²")

    weather = getattr(source_env, "_weather", None)
    if weather is not None:
        wind, turbulence = weather
        lander = getattr(source_env.unwrapped, "lander", None)
        mass = getattr(lander, "mass", None)
        inertia = getattr(lander, "inertia", None)
        if mass:
            lines.append(f"wind a: {abs(float(wind)) / float(mass):.1f} m/s²")
        if inertia:
            lines.append(
                f"turb a: {abs(float(turbulence)) / float(inertia):.1f} rad/s²"
            )

    return lines
