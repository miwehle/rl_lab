"""Simplified Eagle-inspired lander as pure Pygame draw commands.

This is a small design/reference module, not the runtime renderer.
It keeps the recognizable Eagle silhouette while using two side legs so the
shape can later map cleanly to Gymnasium LunarLander's two physical legs.
"""

from __future__ import annotations

from collections.abc import Sequence

import pygame

WIDTH = 220
HEIGHT = 220

Point = tuple[float, float]
RGB = tuple[int, int, int]

SKIN: RGB = (233, 237, 240)
PANEL: RGB = (154, 161, 166)
DARK: RGB = (29, 29, 27)


def draw_simplified_eagle(
    surface: pygame.Surface,
    x: float = 0,
    y: float = 0,
    scale: float = 1.0,
    *,
    skin: RGB = SKIN,
    panel: RGB = PANEL,
    dark: RGB = DARK,
    scale_strokes: bool = True,
) -> None:
    """Draw a compact Eagle-like two-leg lander onto a Pygame surface."""

    stroke = _stroke_width(4, scale, scale_strokes)
    thin = _stroke_width(2, scale, scale_strokes)
    strut = _stroke_width(5, scale, scale_strokes)

    # Ascent stage, inspired by the Eagle cabin and top equipment cluster.
    _polygon(surface, skin, dark, stroke, [(82, 20), (138, 20), (158, 38), (158, 72), (138, 90), (82, 90), (62, 72), (62, 38)], x, y, scale)
    _polygon(surface, panel, dark, thin, [(73, 63), (147, 63), (136, 91), (84, 91)], x, y, scale)
    _polygon(surface, dark, dark, thin, [(86, 41), (110, 32), (134, 41), (121, 59), (99, 59)], x, y, scale)
    _circle(surface, dark, dark, thin, (110, 67), 5, x, y, scale)

    # Descent stage: broad, boxy and deliberately wider than the cabin.
    _rect(surface, skin, dark, stroke, (52, 94, 116, 42), x, y, scale)
    _rect(surface, panel, dark, thin, (70, 104, 80, 22), x, y, scale)
    _rect(surface, skin, dark, thin, (91, 88, 38, 54), x, y, scale)
    _line(surface, dark, thin, (52, 115), (168, 115), x, y, scale)

    # Short engine bell. It must not read as a third landing leg.
    _polygon(surface, dark, dark, thin, [(97, 144), (123, 144), (118, 164), (102, 164)], x, y, scale)

    # Two side legs, mounted far out like the real Eagle but compatible with
    # Gymnasium's two physical leg bodies.
    for side in (-1, 1):
        body_outer = 110 + side * 57
        body_inner = 110 + side * 38
        foot_outer = 110 + side * 83
        foot_inner = 110 + side * 72
        foot_center = 110 + side * 79

        _line(surface, dark, strut, (body_outer, 124), (foot_outer, 183), x, y, scale)
        _line(surface, dark, strut, (body_inner, 136), (foot_inner, 183), x, y, scale)
        _line(surface, dark, thin, (body_outer, 136), (foot_inner, 174), x, y, scale)
        _ellipse(surface, skin, dark, thin, (foot_center - 16, 185, 32, 10), x, y, scale)

    # Minimal antenna, enough to keep the Eagle feel without adding clutter.
    _line(surface, dark, thin, (110, 20), (110, 4), x, y, scale)
    _circle(surface, skin, dark, thin, (110, 4), 4, x, y, scale)


def make_simplified_eagle_surface(scale: float = 1.0, background: RGB | None = None) -> pygame.Surface:
    """Return a new Pygame surface containing the simplified Eagle lander."""

    surface = pygame.Surface((round(WIDTH * scale), round(HEIGHT * scale)), pygame.SRCALPHA)
    if background is not None:
        surface.fill(background)
    draw_simplified_eagle(surface, scale=scale)
    return surface


def save_simplified_eagle_png(
    filename: str = "eagle_simplified_pygame.png",
    *,
    scale: float = 3.0,
    background: RGB = (255, 255, 255),
) -> str:
    """Convenience helper for local scripts and notebooks."""

    pygame.init()
    surface = make_simplified_eagle_surface(scale=scale, background=background)
    pygame.image.save(surface, filename)
    return filename


def _transform(point: Point, x: float, y: float, scale: float) -> tuple[int, int]:
    return (round(x + point[0] * scale), round(y + point[1] * scale))


def _transform_points(points: Sequence[Point], x: float, y: float, scale: float) -> list[tuple[int, int]]:
    return [_transform(point, x, y, scale) for point in points]


def _stroke_width(width: int, scale: float, scale_strokes: bool) -> int:
    return max(1, round(width * scale)) if scale_strokes else max(1, width)


def _polygon(
    surface: pygame.Surface,
    fill: RGB,
    outline: RGB,
    width: int,
    points: Sequence[Point],
    x: float,
    y: float,
    scale: float,
) -> None:
    transformed = _transform_points(points, x, y, scale)
    pygame.draw.polygon(surface, fill, transformed, 0)
    pygame.draw.polygon(surface, outline, transformed, width)


def _rect(
    surface: pygame.Surface,
    fill: RGB,
    outline: RGB,
    width: int,
    rect: tuple[float, float, float, float],
    x: float,
    y: float,
    scale: float,
) -> None:
    rx, ry, rw, rh = rect
    transformed = pygame.Rect(
        round(x + rx * scale),
        round(y + ry * scale),
        round(rw * scale),
        round(rh * scale),
    )
    pygame.draw.rect(surface, fill, transformed, 0)
    pygame.draw.rect(surface, outline, transformed, width)


def _ellipse(
    surface: pygame.Surface,
    fill: RGB,
    outline: RGB,
    width: int,
    rect: tuple[float, float, float, float],
    x: float,
    y: float,
    scale: float,
) -> None:
    rx, ry, rw, rh = rect
    transformed = pygame.Rect(
        round(x + rx * scale),
        round(y + ry * scale),
        round(rw * scale),
        round(rh * scale),
    )
    pygame.draw.ellipse(surface, fill, transformed, 0)
    pygame.draw.ellipse(surface, outline, transformed, width)


def _circle(
    surface: pygame.Surface,
    fill: RGB,
    outline: RGB,
    width: int,
    center: Point,
    radius: float,
    x: float,
    y: float,
    scale: float,
) -> None:
    transformed = _transform(center, x, y, scale)
    scaled_radius = max(1, round(radius * scale))
    pygame.draw.circle(surface, fill, transformed, scaled_radius, 0)
    pygame.draw.circle(surface, outline, transformed, scaled_radius, width)


def _line(
    surface: pygame.Surface,
    color: RGB,
    width: int,
    start: Point,
    end: Point,
    x: float,
    y: float,
    scale: float,
) -> None:
    pygame.draw.line(surface, color, _transform(start, x, y, scale), _transform(end, x, y, scale), width)
