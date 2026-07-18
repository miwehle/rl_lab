"""Colored Eagle skin for LunarLander-compatible rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TypeAlias

from gymnasium.envs.box2d import lunar_lander

Point: TypeAlias = tuple[float, float]

_ASSET_DIR = Path(__file__).resolve().parents[1] / "_skin_assets" / "eagle_colored"
_BODY_PNG = _ASSET_DIR / "eagle_colored_body.png"
_SIDE_LEGS_PNG = _ASSET_DIR / "eagle_colored_side_legs.png"
_LANDING_Y_OFFSET = 50.0


@dataclass(frozen=True)
class ColoredEagleSkin:
    """Draw the colored Eagle body and Gym-coupled side legs from PNG assets."""

    scale: float = 0.16
    body_anchor: Point = (251.0, 282.0 + _LANDING_Y_OFFSET)
    right_leg_rest_angle: float = 0.492
    left_leg_rest_angle: float = -0.492

    def draw(self, surface, env, *, render_scale: int = 1) -> None:
        """Draw the skin on an already screen-oriented LunarLander surface."""
        if getattr(env, "lander", None) is None or len(getattr(env, "legs", ())) < 2:
            return

        body, side_legs = _assets()
        scale = self.scale * render_scale
        _blit_on_body(surface, side_legs.surface, env.lander, side_legs.anchor, scale, render_scale=render_scale)
        _blit_on_body(surface, body.surface, env.lander, self.body_anchor, scale, render_scale=render_scale)


@dataclass(frozen=True)
class _Asset:
    surface: object
    anchor: Point


_CACHED_ASSETS: tuple[_Asset, _Asset] | None = None


def _assets() -> tuple[_Asset, _Asset]:
    global _CACHED_ASSETS
    if _CACHED_ASSETS is None:
        import pygame

        body = pygame.image.load(str(_BODY_PNG))
        side_legs = pygame.image.load(str(_SIDE_LEGS_PNG))
        _CACHED_ASSETS = (
            _Asset(body, (251.0, 282.0 + _LANDING_Y_OFFSET)),
            _Asset(side_legs, (251.0, 282.0 + _LANDING_Y_OFFSET)),
        )
    return _CACHED_ASSETS


def _blit_on_body(
    surface, image, body, source_anchor: Point, scale: float, angle_offset: float = 0.0, *, render_scale: int = 1
) -> None:
    import pygame

    scaled_size = (
        max(1, round(image.get_width() * scale)),
        max(1, round(image.get_height() * scale)),
    )
    scaled = pygame.transform.smoothscale(image, scaled_size)
    scaled_anchor = (source_anchor[0] * scale, source_anchor[1] * scale)
    angle_rad = float(body.angle) - angle_offset
    if abs(angle_rad) < math.radians(0.5):
        angle_rad = 0.0
    angle = math.degrees(angle_rad)
    rotated = pygame.transform.rotate(scaled, angle)
    rotated_anchor = _rotated_anchor(scaled.get_size(), rotated.get_size(), scaled_anchor, angle)
    target_anchor = _body_to_screen(body.position, render_scale=render_scale)
    surface.blit(rotated, (round(target_anchor[0] - rotated_anchor[0]), round(target_anchor[1] - rotated_anchor[1])))


def _rotated_anchor(
    source_size: tuple[int, int], rotated_size: tuple[int, int], anchor: Point, angle_degrees: float
) -> Point:
    source_center = (source_size[0] / 2, source_size[1] / 2)
    rotated_center = (rotated_size[0] / 2, rotated_size[1] / 2)
    dx = anchor[0] - source_center[0]
    dy = anchor[1] - source_center[1]
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return rotated_center[0] + dx * cos_a - dy * sin_a, rotated_center[1] + dx * sin_a + dy * cos_a


def _body_to_screen(position, *, render_scale: int = 1) -> tuple[int, int]:
    return round(position[0] * lunar_lander.SCALE * render_scale), round(
        (lunar_lander.VIEWPORT_H - position[1] * lunar_lander.SCALE) * render_scale
    )
