"""Custom rendering colors for Gymnasium LunarLander videos."""

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Protocol

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
_WIND_ACCEL_SCALE = 4.0
_TURBULENCE_ACCEL_SCALE = 2.4
_WIND_COLOR_STOPS = (
    (0.0, (72, 220, 92)),
    (1.0, (255, 230, 72)),
    (2.0, (255, 146, 43)),
    (3.0, (255, 55, 55)),
)
_TURBULENCE_COLOR_STOPS = (
    (0.0, (72, 220, 92)),
    (0.7, (255, 230, 72)),
    (1.4, (255, 146, 43)),
    (2.1, (255, 55, 55)),
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


class LanderSkin(Protocol):
    """Optional visual replacement for the default Gym lander body."""

    def draw(self, surface, env) -> None:
        """Draw the lander on an already screen-oriented surface."""


_WORLD_COLORS = {
    "earth": LanderColors(sky=(143, 199, 232), ground=(111, 127, 82), ground_outline=(111, 127, 82)),
    "venus": LanderColors(sky=(214, 168, 92), ground=(138, 103, 64), ground_outline=(138, 103, 64)),
    "mars": LanderColors(sky=(201, 149, 122), ground=(139, 79, 61), ground_outline=(139, 79, 61)),
    "moon": LanderColors(sky=(5, 7, 10), ground=(119, 122, 125), ground_outline=(119, 122, 125)),
    "mercury": LanderColors(sky=(5, 7, 10), ground=(111, 107, 100), ground_outline=(111, 107, 100)),
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
        skin: LanderSkin | None = None,
    ) -> None:
        super().__init__(env)
        self.colors = colors or LanderColors()
        self.overlay = overlay
        self.skin = skin
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
            scaled_poly.append((coord[0] * lunar_lander.SCALE, coord[1] * lunar_lander.SCALE))
        pygame.draw.polygon(env.surf, colors.sky, scaled_poly)
        gfxdraw.aapolygon(env.surf, scaled_poly, colors.sky)
        pygame.draw.aaline(env.surf, colors.ground_outline, scaled_poly[0], scaled_poly[1])

    for obj in env.particles + env.drawlist:
        if skin is not None and _is_lander_body(obj, env):
            continue
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
                pygame.draw.aalines(env.surf, color=outline, points=path, closed=True)

    _draw_flags(env.surf, env, colors, gfxdraw)

    env.surf = pygame.transform.flip(env.surf, False, True)
    if skin is not None:
        skin.draw(env.surf, env)
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
        return np.transpose(np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2))

    return None


def _object_colors(obj, env, colors: LanderColors) -> tuple[RGB, RGB]:
    if _is_lander_body(obj, env):
        return colors.lander_fill, colors.lander_outline
    return obj.color1, obj.color2


def _is_lander_body(obj, env) -> bool:
    return obj is env.lander or any(obj is leg for leg in env.legs)


def _draw_flags(surface, env, colors: LanderColors, gfxdraw) -> None:
    import pygame

    wind = _wind_strength(env)
    for x in [env.helipad_x1, env.helipad_x2]:
        x = x * lunar_lander.SCALE
        flagy1 = env.helipad_y * lunar_lander.SCALE
        flagy2 = flagy1 + 50
        pygame.draw.line(surface, color=colors.flag_pole, start_pos=(x, flagy1), end_pos=(x, flagy2), width=1)
        _draw_windsock(surface, gfxdraw, (x, flagy2 - 8), wind)


def _wind_strength(env) -> tuple[float, float]:
    env = getattr(env, "unwrapped", env)
    if not getattr(env, "enable_wind", False):
        return 0.0, 0.0
    wind_idx = getattr(env, "wind_idx", None)
    if wind_idx is None:
        return 0.0, 0.0
    wind_power = float(getattr(env, "wind_power", 0.0))
    wind = _force_wave(wind_idx) * wind_power
    return wind, abs(wind_power)


def _draw_windsock(surface, gfxdraw, anchor: tuple[float, float], wind: tuple[float, float]) -> None:
    import pygame

    wind_value, max_wind = wind
    direction = 1 if wind_value >= 0 else -1
    strength = _clamp_float(abs(wind_value) / max_wind if max_wind else 0.0, 0.0, 1.0)
    length = 18 + 28 * strength
    droop = 1.0 - strength
    segment_count = 5
    centerline = []
    for index in range(segment_count + 1):
        t = index / segment_count
        x = anchor[0] + direction * length * t
        y = anchor[1] + 4 + 18 * droop * (t**1.4)
        centerline.append((x, y))

    widths = [10 - 5 * index / segment_count for index in range(segment_count + 1)]
    colors = ((220, 0, 0), (245, 245, 245), (220, 0, 0), (245, 245, 245), (220, 0, 0))
    for index in range(segment_count):
        p0 = centerline[index]
        p1 = centerline[index + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = math.hypot(dx, dy) or 1.0
        nx = -dy / norm
        ny = dx / norm
        w0 = widths[index] / 2
        w1 = widths[index + 1] / 2
        points = [
            (p0[0] + nx * w0, p0[1] + ny * w0),
            (p1[0] + nx * w1, p1[1] + ny * w1),
            (p1[0] - nx * w1, p1[1] - ny * w1),
            (p0[0] - nx * w0, p0[1] - ny * w0),
        ]
        pygame.draw.polygon(surface, colors[index], points)
        gfxdraw.aapolygon(surface, points, colors[index])
        pygame.draw.aalines(surface, (0, 0, 0), True, points)


def _draw_overlay(surface, source_env, overlay: LanderOverlay) -> None:
    import pygame

    lines = _overlay_lines(source_env)

    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, 18)
    score = getattr(source_env, "score", None)
    if score is not None:
        score_text = f"score: {float(score):.1f}"
        _draw_text(surface, font, score_text, (surface.get_width() - 116, 8), overlay)

    x, y = 8, 8
    line_height = font.get_linesize()
    for index, (line, direction) in enumerate(lines):
        position = (x, y + index * line_height)
        text = _draw_text(surface, font, line, position, overlay)
        if direction is not None:
            arrow_center = (position[0] + text.get_width() + 12, position[1] + line_height // 2)
            _draw_arrow(surface, arrow_center, direction, overlay)

    env = getattr(source_env, "env", source_env)
    _draw_wind_indicator(surface, font, env, overlay)
    _draw_turbulence_indicator(surface, font, env, overlay)


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
            lines.append((f"wind max: {abs(float(wind)) / float(mass):.1f} m/s²", None))
        if inertia:
            turbulence_degrees = math.degrees(abs(float(turbulence)) / float(inertia))
            lines.append((f"turb max: {turbulence_degrees:.0f}°/s²", None))

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


def _draw_wind_indicator(surface, font, env, overlay: LanderOverlay) -> None:
    wind = _wind_acceleration(env)
    if wind is None:
        return

    acceleration, _ = wind
    center = (surface.get_width() // 2, 26)
    color = _force_color(abs(acceleration), _WIND_COLOR_STOPS)
    _draw_horizontal_force_arrow(
        surface,
        center,
        acceleration,
        scale=_WIND_ACCEL_SCALE,
        color=color,
        shadow_color=overlay.shadow_color,
    )
    _draw_centered_text(surface, font, f"wind {acceleration:+.1f} m/s²", (center[0], center[1] + 13), color, overlay)


def _draw_turbulence_indicator(surface, font, env, overlay: LanderOverlay) -> None:
    turbulence = _turbulence_acceleration(env)
    lander_center = _lander_screen_position(env)
    if turbulence is None or lander_center is None:
        return

    acceleration, _ = turbulence
    color = _force_color(abs(acceleration), _TURBULENCE_COLOR_STOPS)
    if lander_center[1] < 80:
        center = (_clamp(lander_center[0] + 110, 26, surface.get_width() - 26), 62)
    else:
        center = (
            _clamp(lander_center[0] + 92, 26, surface.get_width() - 26),
            _clamp(lander_center[1] - 38, 26, surface.get_height() - 26),
        )
    _draw_torque_arrow(
        surface,
        center,
        acceleration,
        scale=_TURBULENCE_ACCEL_SCALE,
        color=color,
        shadow_color=overlay.shadow_color,
    )
    _draw_centered_text(
        surface,
        font,
        f"turb {math.degrees(acceleration):+.0f}°/s²",
        (center[0], center[1] + 18),
        color,
        overlay,
    )


def _draw_centered_text(
    surface, font, line: str, center: tuple[int, int], color: RGB, overlay: LanderOverlay
) -> None:
    shadow = font.render(line, True, overlay.shadow_color)
    text = font.render(line, True, color)
    position = (center[0] - text.get_width() // 2, center[1])
    surface.blit(shadow, (position[0] + 1, position[1] + 1))
    surface.blit(text, position)


def _wind_acceleration(env) -> tuple[float, float] | None:
    env = getattr(env, "unwrapped", env)
    lander = getattr(env, "lander", None)
    mass = getattr(lander, "mass", None)
    wind_idx = getattr(env, "wind_idx", None)
    if not getattr(env, "enable_wind", False) or lander is None or not mass or wind_idx is None:
        return None

    wind_power = float(getattr(env, "wind_power", 0.0))
    max_acceleration = _max_acceleration(wind_power, mass)
    if _has_ground_contact(env):
        return 0.0, max_acceleration
    return _force_wave(wind_idx) * wind_power / float(mass), max_acceleration


def _turbulence_acceleration(env) -> tuple[float, float] | None:
    env = getattr(env, "unwrapped", env)
    lander = getattr(env, "lander", None)
    inertia = getattr(lander, "inertia", None)
    torque_idx = getattr(env, "torque_idx", None)
    if not getattr(env, "enable_wind", False) or lander is None or not inertia or torque_idx is None:
        return None

    turbulence_power = float(getattr(env, "turbulence_power", 0.0))
    max_acceleration = _max_acceleration(turbulence_power, inertia)
    if _has_ground_contact(env):
        return 0.0, max_acceleration
    return _force_wave(torque_idx) * turbulence_power / float(inertia), max_acceleration


def _force_wave(index) -> float:
    return math.tanh(math.sin(0.02 * int(index)) + math.sin(math.pi * 0.01 * int(index)))


def _max_acceleration(force: float, divisor: float) -> float:
    if not divisor:
        return 0.0
    return abs(float(force)) / float(divisor)


def _has_ground_contact(env) -> bool:
    return any(getattr(leg, "ground_contact", False) for leg in getattr(env, "legs", ()))


def _force_color(value: float, stops: tuple[tuple[float, RGB], ...]) -> RGB:
    if value <= stops[0][0]:
        return stops[0][1]
    for (lower_value, lower_color), (upper_value, upper_color) in zip(stops, stops[1:]):
        if value <= upper_value:
            ratio = (value - lower_value) / (upper_value - lower_value)
            return _interpolate_color(lower_color, upper_color, ratio)
    return stops[-1][1]


def _interpolate_color(lower: RGB, upper: RGB, ratio: float) -> RGB:
    return tuple(round(low + (high - low) * ratio) for low, high in zip(lower, upper))


def _lander_screen_position(env) -> tuple[int, int] | None:
    env = getattr(env, "unwrapped", env)
    lander = getattr(env, "lander", None)
    if lander is None:
        return None
    return (
        int(lander.position.x * lunar_lander.SCALE),
        int(lunar_lander.VIEWPORT_H - lander.position.y * lunar_lander.SCALE),
    )


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


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
    surface, center: tuple[int, int], direction: tuple[float, float], overlay: LanderOverlay
) -> None:
    shadow_center = (center[0] + 1, center[1] + 1)
    _draw_arrow_shape(surface, shadow_center, direction, overlay.shadow_color)
    _draw_arrow_shape(surface, center, direction, overlay.text_color)


def _draw_arrow_shape(surface, center: tuple[int, int], direction: tuple[float, float], color: RGB) -> None:
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


def _draw_horizontal_force_arrow(
    surface,
    center: tuple[int, int],
    value: float,
    *,
    scale: float,
    color: RGB,
    shadow_color: RGB,
) -> None:
    length = 14 + min(abs(value) / scale, 1.0) * 96
    direction = 1 if value >= 0 else -1
    start = (center[0] - direction * length / 2, center[1])
    end = (center[0] + direction * length / 2, center[1])
    _draw_line_arrow(surface, (start[0] + 1, start[1] + 1), (end[0] + 1, end[1] + 1), shadow_color, width=4)
    _draw_line_arrow(surface, start, end, color, width=3)


def _draw_torque_arrow(
    surface,
    center: tuple[int, int],
    value: float,
    *,
    scale: float,
    color: RGB,
    shadow_color: RGB,
) -> None:
    import pygame

    radius = 20
    strength = min(abs(value) / scale, 1.0)
    if strength == 0:
        return
    span = math.radians(strength * 270)
    start_angle = math.radians(180)
    direction = 1 if value >= 0 else -1
    point_count = 22
    points = [
        (
            center[0] + math.cos(start_angle + direction * span * index / (point_count - 1)) * radius,
            center[1] + math.sin(start_angle + direction * span * index / (point_count - 1)) * radius,
        )
        for index in range(point_count)
    ]
    width = 2 + round(strength * 2)
    shadow_points = [(x + 1, y + 1) for x, y in points]
    pygame.draw.lines(surface, shadow_color, False, shadow_points, width + 1)
    pygame.draw.lines(surface, color, False, points, width)

    end_angle = start_angle + direction * span
    tangent = (-math.sin(end_angle) * direction, math.cos(end_angle) * direction)
    tip = (points[-1][0] + tangent[0] * 2, points[-1][1] + tangent[1] * 2)
    _draw_arrow_head(surface, (tip[0] + 1, tip[1] + 1), tangent, shadow_color, size=9)
    _draw_arrow_head(surface, tip, tangent, color, size=8)


def _draw_line_arrow(surface, start: tuple[float, float], end: tuple[float, float], color: RGB, *, width: int) -> None:
    import pygame

    pygame.draw.line(surface, color, start, end, width)
    direction = (end[0] - start[0], end[1] - start[1])
    _draw_arrow_head(surface, end, direction, color, size=8)


def _draw_arrow_head(surface, tip: tuple[float, float], direction: tuple[float, float], color: RGB, *, size: int) -> None:
    import pygame

    dx, dy = direction
    length = math.hypot(dx, dy)
    if not length:
        return
    angle = math.atan2(dy, dx)
    left = (
        tip[0] - size * math.cos(angle - math.pi / 6),
        tip[1] - size * math.sin(angle - math.pi / 6),
    )
    right = (
        tip[0] - size * math.cos(angle + math.pi / 6),
        tip[1] - size * math.sin(angle + math.pi / 6),
    )
    pygame.draw.polygon(surface, color, [tip, left, right])
