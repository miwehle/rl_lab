"""HUD-style overlay composition for SolarSystemLander videos.

The overlay is a head-up display (HUD): information drawn over the scene without
being part of the world geometry, such as score, conditions text, and analysis
vectors.
"""

import math

from hpo.evaluation.rendering.solar_system_lander._colors import LanderOverlay
from hpo.evaluation.rendering.solar_system_lander._env_state import EnvState
from hpo.evaluation.rendering.solar_system_lander._overlay.vectors import (
    disturbance_vector_origin,
    draw_kick_indicator,
    draw_turbulence_indicator,
    draw_wind_indicator,
)


def draw_overlay(surface, env_state: EnvState, overlay: LanderOverlay, *, render_scale: int = 1) -> None:
    import pygame

    lines = _overlay_lines(env_state)

    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, 18 * render_scale)
    if env_state.score is not None:
        score_text = f"score: {float(env_state.score):.0f}"
        _draw_text(surface, font, score_text, (surface.get_width() - 116 * render_scale, 8 * render_scale), overlay)

    x, y = 8 * render_scale, 8 * render_scale
    line_height = font.get_linesize()
    for index, line in enumerate(lines):
        position = (x, y + index * line_height)
        _draw_text(surface, font, line, position, overlay)

    vector_origin = disturbance_vector_origin(surface, render_scale=render_scale)
    draw_wind_indicator(surface, font, env_state.wind, overlay, vector_origin, render_scale=render_scale)
    draw_kick_indicator(surface, font, env_state, overlay, vector_origin, render_scale=render_scale)
    draw_turbulence_indicator(surface, font, env_state.turbulence, env_state.lander_screen_position, overlay, render_scale=render_scale)


def _draw_text(surface, font, line: str, position: tuple[int, int], overlay: LanderOverlay):
    shadow = font.render(line, True, overlay.shadow_color)
    text = font.render(line, True, overlay.text_color)
    shadow_offset = max(1, round(font.get_height() / 18))
    surface.blit(shadow, (position[0] + shadow_offset, position[1] + shadow_offset))
    surface.blit(text, position)
    return text


def _overlay_lines(env_state: EnvState) -> list[str]:
    lines = []
    if env_state.world_name is not None:
        lines.append(env_state.world_name.title())
    if env_state.gravity is not None:
        lines.append(f"g: {abs(env_state.gravity):.1f} m/s²")
    if env_state.wind.max_acceleration is not None:
        lines.append(f"wind max: {env_state.wind.max_acceleration:.1f} m/s²")
    if env_state.turbulence.max_acceleration is not None:
        lines.append(f"turb max: {math.degrees(env_state.turbulence.max_acceleration):.0f}°/s²")
    if env_state.initial_kick_delta_v is not None:
        lines.append(f"kick: {math.hypot(*env_state.initial_kick_delta_v):.1f} m/s")
    return lines
