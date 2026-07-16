"""Colored Eagle SVG skin for LunarLander-compatible rendering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TypeAlias

from gymnasium.envs.box2d import lunar_lander

Point: TypeAlias = tuple[float, float]

_ASSET_DIR = Path(__file__).resolve().parents[1] / "_skin_assets" / "eagle_colored"
_BODY_SVG = _ASSET_DIR / "eagle_colored_body.svg"
_SIDE_LEGS_SVG = _ASSET_DIR / "eagle_colored_side_legs.svg"
_LEG_ANCHOR_Y_OFFSET = 18.75


@dataclass(frozen=True)
class ColoredEagleSkin:
    """Draw the colored Eagle body and Gym-coupled side legs from SVG assets."""

    scale: float = 0.16
    body_anchor: Point = (251.0, 282.0)
    right_leg_rest_angle: float = 0.492
    left_leg_rest_angle: float = -0.492

    def draw(self, surface, env) -> None:
        """Draw the skin on an already screen-oriented LunarLander surface."""
        if getattr(env, "lander", None) is None or len(getattr(env, "legs", ())) < 2:
            return

        body, left_leg, right_leg = _assets()
        _blit_on_body(surface, body.surface, env.lander, self.body_anchor, self.scale)
        _blit_on_body(
            surface,
            left_leg.surface,
            env.legs[1],
            left_leg.anchor,
            self.scale,
            angle_offset=self.left_leg_rest_angle,
        )
        _blit_on_body(
            surface,
            right_leg.surface,
            env.legs[0],
            right_leg.anchor,
            self.scale,
            angle_offset=self.right_leg_rest_angle,
        )


@dataclass(frozen=True)
class _Asset:
    surface: object
    anchor: Point


_CACHED_ASSETS: tuple[_Asset, _Asset, _Asset] | None = None


def _assets() -> tuple[_Asset, _Asset, _Asset]:
    global _CACHED_ASSETS
    if _CACHED_ASSETS is None:
        import pygame

        body = pygame.image.load(str(_BODY_SVG))
        side_legs = pygame.image.load(str(_SIDE_LEGS_SVG))
        left_leg = _crop_half(side_legs, left=True)
        right_leg = _crop_half(side_legs, left=False)
        _CACHED_ASSETS = (
            _Asset(body, (251.0, 282.0)),
            left_leg,
            right_leg,
        )
    return _CACHED_ASSETS


def _crop_half(surface, *, left: bool) -> _Asset:
    import pygame

    half_width = surface.get_width() // 2
    half_rect = pygame.Rect(0 if left else half_width, 0, half_width, surface.get_height())
    bbox = surface.subsurface(half_rect).get_bounding_rect(1)
    if bbox.width <= 0 or bbox.height <= 0:
        raise ValueError("colored Eagle side-leg SVG does not contain visible pixels")

    rect = pygame.Rect(half_rect.x + bbox.x, bbox.y, bbox.width, bbox.height)
    cropped = pygame.Surface(rect.size, pygame.SRCALPHA)
    cropped.blit(surface, (0, 0), rect)
    full_anchor = (rect.centerx, rect.centery + _LEG_ANCHOR_Y_OFFSET)
    return _Asset(cropped, (full_anchor[0] - rect.x, full_anchor[1] - rect.y))


def _blit_on_body(surface, image, body, source_anchor: Point, scale: float, angle_offset: float = 0.0) -> None:
    import pygame

    scaled_size = (
        max(1, round(image.get_width() * scale)),
        max(1, round(image.get_height() * scale)),
    )
    scaled = pygame.transform.smoothscale(image, scaled_size)
    scaled_anchor = (source_anchor[0] * scale, source_anchor[1] * scale)
    angle = math.degrees(float(body.angle) - angle_offset)
    rotated = pygame.transform.rotate(scaled, angle)
    rotated_anchor = _rotated_anchor(scaled.get_size(), rotated.get_size(), scaled_anchor, angle)
    target_anchor = _body_to_screen(body.position)
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


def _body_to_screen(position) -> tuple[int, int]:
    return round(position[0] * lunar_lander.SCALE), round(
        lunar_lander.VIEWPORT_H - position[1] * lunar_lander.SCALE
    )
