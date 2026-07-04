"""Custom rendering colors for Gymnasium LunarLander videos."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d import lunar_lander
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

RGB = tuple[int, int, int]
_KICK_DIRECTIONS = (
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 0.0),
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
)


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


_WORLD_COLORS = {
    "earth": LanderColors(
        sky=(143, 199, 232),
        ground=(111, 127, 82),
        ground_outline=(111, 127, 82),
    ),
    "venus": LanderColors(
        sky=(214, 168, 92),
        ground=(138, 103, 64),
        ground_outline=(138, 103, 64),
    ),
    "mars": LanderColors(
        sky=(201, 149, 122),
        ground=(139, 79, 61),
        ground_outline=(139, 79, 61),
    ),
    "moon": LanderColors(
        sky=(5, 7, 10),
        ground=(119, 122, 125),
        ground_outline=(119, 122, 125),
    ),
    "mercury": LanderColors(
        sky=(5, 7, 10),
        ground=(111, 107, 100),
        ground_outline=(111, 107, 100),
    ),
}


def world_colors(worlds: Iterable[str]) -> list[LanderColors]:
    """Return render colors in the same order as the requested worlds."""
    colors = []
    for world in worlds:
        name = str(world)
        try:
            colors.append(_WORLD_COLORS[name])
        except KeyError:
            raise ValueError(f"unknown world color: {name}") from None
    return colors


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
        self.reset_seed: int | None = None
        self.score = 0.0

    def reset(self, *, seed: int | None = None, options=None):
        self.reset_seed = seed
        self.score = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.score += float(reward)
        return observation, reward, terminated, truncated, info

    def render(self):
        return _render_lunar_lander(
            self.env.unwrapped,
            self,
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
    score = getattr(source_env, "score", None)
    if score is not None:
        score_text = f"score: {float(score):.1f}"
        _draw_text(
            surface,
            font,
            score_text,
            (surface.get_width() - 116, 8),
            overlay,
        )

    x, y = 8, 8
    line_height = font.get_linesize()
    for index, (line, direction) in enumerate(lines):
        position = (x, y + index * line_height)
        text = _draw_text(surface, font, line, position, overlay)
        if direction is not None:
            arrow_center = (
                position[0] + text.get_width() + 12,
                position[1] + line_height // 2,
            )
            _draw_arrow(surface, arrow_center, direction, overlay)


def _overlay_lines(source_env) -> list[tuple[str, tuple[float, float] | None]]:
    lines = []
    env = getattr(source_env, "env", source_env)
    world = getattr(env, "world", None)
    if world is not None:
        name = getattr(world, "name", None)
        gravity = getattr(world, "gravity", None)
        if name is not None:
            lines.append((str(name).title(), None))
        if gravity is not None:
            lines.append((f"g: {abs(float(gravity)):.1f} m/s²", None))

    lander = getattr(env.unwrapped, "lander", None)
    mass = getattr(lander, "mass", None)
    inertia = getattr(lander, "inertia", None)
    weather = getattr(env, "_weather", None)
    if weather is not None:
        wind, turbulence = weather
        if mass:
            lines.append((f"wind: {abs(float(wind)) / float(mass):.1f} m/s²", None))
        if inertia:
            lines.append(
                (
                    f"turb: {abs(float(turbulence)) / float(inertia):.1f} rad/s²",
                    None,
                )
            )

    kick = _initial_kick(getattr(source_env, "reset_seed", None), mass)
    if kick is not None:
        delta_v, direction = kick
        lines.append((f"kick: {delta_v:.1f} m/s", direction))

    return lines


def _draw_text(surface, font, line: str, position: tuple[int, int], overlay: LanderOverlay):
    shadow = font.render(line, True, overlay.shadow_color)
    text = font.render(line, True, overlay.text_color)
    surface.blit(shadow, (position[0] + 1, position[1] + 1))
    surface.blit(text, position)
    return text


def _initial_kick(seed: int | None, mass) -> tuple[float, tuple[float, float]] | None:
    if seed is None or not mass:
        return None
    fx, fy = _initial_force(seed)
    delta_v = float(np.hypot(fx, fy)) * (1.0 / lunar_lander.FPS) / float(mass)
    return delta_v, _kick_direction(fx, fy)


def _kick_direction(fx: float, fy: float) -> tuple[float, float]:
    index = round(math.atan2(fy, fx) / (math.pi / 4)) % len(_KICK_DIRECTIONS)
    return _KICK_DIRECTIONS[index]


def _initial_force(seed: int) -> tuple[float, float]:
    rng, _ = seeding.np_random(seed)
    viewport_height = lunar_lander.VIEWPORT_H / lunar_lander.SCALE
    chunks = 11
    rng.uniform(0, viewport_height / 2, size=(chunks + 1,))
    fx = rng.uniform(-lunar_lander.INITIAL_RANDOM, lunar_lander.INITIAL_RANDOM)
    fy = rng.uniform(-lunar_lander.INITIAL_RANDOM, lunar_lander.INITIAL_RANDOM)
    return float(fx), float(fy)


def _draw_arrow(
    surface,
    center: tuple[int, int],
    direction: tuple[float, float],
    overlay: LanderOverlay,
) -> None:
    shadow_center = (center[0] + 1, center[1] + 1)
    _draw_arrow_shape(surface, shadow_center, direction, overlay.shadow_color)
    _draw_arrow_shape(surface, center, direction, overlay.text_color)


def _draw_arrow_shape(
    surface,
    center: tuple[int, int],
    direction: tuple[float, float],
    color: RGB,
) -> None:
    import pygame

    dx, dy = direction
    length = math.hypot(dx, dy)
    if not length:
        return
    dx /= length
    dy = -dy / length
    arrow_length = 12
    head_length = 4
    start = (center[0] - dx * arrow_length / 2, center[1] - dy * arrow_length / 2)
    end = (center[0] + dx * arrow_length / 2, center[1] + dy * arrow_length / 2)
    angle = math.atan2(dy, dx)
    left = (
        end[0] - head_length * math.cos(angle - math.pi / 6),
        end[1] - head_length * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head_length * math.cos(angle + math.pi / 6),
        end[1] - head_length * math.sin(angle + math.pi / 6),
    )
    pygame.draw.line(surface, color, start, end, 2)
    pygame.draw.polygon(surface, color, [end, left, right])
