"""Vector and torque visualizations for SolarSystemLander overlays."""

import math

from hpo.evaluation.rendering.solar_system_lander.colors import (
    KICK_COLOR_STOPS,
    LanderOverlay,
    RGB,
    TURBULENCE_COLOR_STOPS,
    WIND_COLOR_STOPS,
    _force_color,
)
from hpo.evaluation.rendering.solar_system_lander.env_state import EnvState, TurbulenceState, WindState

_WIND_ACCEL_SCALE = 4.0
_TURBULENCE_ACCEL_SCALE = 2.4
_KICK_DELTA_V_SCALE = 5.5
_KICK_VISIBLE_STEPS = 25
_DISTURBANCE_VECTOR_ORIGIN_Y = 76


def draw_wind_indicator(
    surface, font, wind: WindState, overlay: LanderOverlay, origin: tuple[int, int]
) -> None:
    if wind.acceleration is None:
        return

    acceleration = wind.acceleration
    color = _force_color(abs(acceleration), WIND_COLOR_STOPS)
    _draw_horizontal_force_arrow(
        surface,
        origin,
        acceleration,
        scale=_WIND_ACCEL_SCALE,
        color=color,
        shadow_color=overlay.shadow_color,
    )
    _draw_centered_text(surface, font, f"wind {acceleration:+.1f} m/s²", (origin[0], origin[1] + 13), color, overlay)


def draw_kick_indicator(
    surface,
    font,
    env_state: EnvState,
    overlay: LanderOverlay,
    origin: tuple[int, int],
) -> None:
    if env_state.step >= _KICK_VISIBLE_STEPS:
        return
    if env_state.initial_kick_delta_v is None:
        return

    delta_v = math.hypot(*env_state.initial_kick_delta_v)
    color = _force_color(delta_v, KICK_COLOR_STOPS)
    length = 14 + min(delta_v / _KICK_DELTA_V_SCALE, 1.0) * 96
    screen_direction = _normalized((env_state.initial_kick_delta_v[0], -env_state.initial_kick_delta_v[1]))
    length = _fit_vector_length(surface, origin, screen_direction, length, margin=6)
    end = (origin[0] + screen_direction[0] * length, origin[1] + screen_direction[1] * length)
    _draw_line_arrow(
        surface,
        (origin[0] + 1, origin[1] + 1),
        (end[0] + 1, end[1] + 1),
        overlay.shadow_color,
        width=4,
    )
    _draw_line_arrow(surface, origin, end, color, width=3)
    _draw_centered_text(surface, font, f"kick {delta_v:.1f} m/s", (origin[0], origin[1] - 20), color, overlay)


def disturbance_vector_origin(surface) -> tuple[int, int]:
    return surface.get_width() // 2, _DISTURBANCE_VECTOR_ORIGIN_Y


def draw_turbulence_indicator(
    surface, font, turbulence: TurbulenceState, lander_center: tuple[int, int] | None, overlay: LanderOverlay
) -> None:
    acceleration = turbulence.acceleration
    if acceleration is None or lander_center is None:
        return

    color = _force_color(abs(acceleration), TURBULENCE_COLOR_STOPS)
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


def _normalized(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    if not length:
        return 0.0, 0.0
    return vector[0] / length, vector[1] / length


def _fit_vector_length(
    surface,
    origin: tuple[int, int],
    direction: tuple[float, float],
    length: float,
    *,
    margin: int,
) -> float:
    max_length = length
    if direction[0] > 0:
        max_length = min(max_length, (surface.get_width() - margin - origin[0]) / direction[0])
    elif direction[0] < 0:
        max_length = min(max_length, (origin[0] - margin) / -direction[0])
    if direction[1] > 0:
        max_length = min(max_length, (surface.get_height() - margin - origin[1]) / direction[1])
    elif direction[1] < 0:
        max_length = min(max_length, (origin[1] - margin) / -direction[1])
    return max(0.0, max_length)


def _draw_horizontal_force_arrow(
    surface,
    origin: tuple[int, int],
    value: float,
    *,
    scale: float,
    color: RGB,
    shadow_color: RGB,
) -> None:
    length = 14 + min(abs(value) / scale, 1.0) * 96
    direction = 1 if value >= 0 else -1
    end = (origin[0] + direction * length, origin[1])
    _draw_line_arrow(surface, (origin[0] + 1, origin[1] + 1), (end[0] + 1, end[1] + 1), shadow_color, width=4)
    _draw_line_arrow(surface, origin, end, color, width=3)


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
    direction_length = math.hypot(*direction)
    if direction_length:
        tip = (end[0] + direction[0] / direction_length * 2, end[1] + direction[1] / direction_length * 2)
    else:
        tip = end
    _draw_arrow_head(surface, tip, direction, color, size=8)


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


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))
