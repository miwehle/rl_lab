"""Windsock drawing for SolarSystemLander videos."""

import math

from gymnasium.envs.box2d import lunar_lander

from hpo.evaluation.rendering.solar_system_lander.colors import LanderColors
from hpo.evaluation.rendering.solar_system_lander.env_state import WindState

_WINDSOCK_DEAD_ZONE = 0.4
_WINDSOCK_FULL_ACCEL = 2.0


def draw_flags(surface, env, colors: LanderColors, wind: WindState, gfxdraw) -> None:
    import pygame

    wind = (wind.windsock_acceleration, wind.windsock_max_acceleration)
    for x in [env.helipad_x1, env.helipad_x2]:
        x = x * lunar_lander.SCALE
        flagy1 = env.helipad_y * lunar_lander.SCALE
        flagy2 = flagy1 + 50
        pygame.draw.line(surface, color=colors.flag_pole, start_pos=(x, flagy1), end_pos=(x, flagy2), width=1)
        _draw_windsock(surface, gfxdraw, (x, flagy2 - 8), wind)


def _draw_windsock(surface, gfxdraw, anchor: tuple[float, float], wind: tuple[float, float]) -> None:
    import pygame

    wind_acceleration, _ = wind
    direction = 1 if wind_acceleration >= 0 else -1
    strength = _windsock_strength(abs(wind_acceleration))
    length = 16 + 30 * strength
    horizontal = strength
    droop = 1.0 - strength
    segment_count = 5
    centerline = []
    for index in range(segment_count + 1):
        t = index / segment_count
        x = anchor[0] + direction * length * horizontal * t
        y = anchor[1] + 4 - (6 + 22 * droop) * (t**1.25)
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


def _windsock_strength(wind_acceleration: float) -> float:
    return _clamp_float(
        (wind_acceleration - _WINDSOCK_DEAD_ZONE) / (_WINDSOCK_FULL_ACCEL - _WINDSOCK_DEAD_ZONE),
        0.0,
        1.0,
    )


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))
