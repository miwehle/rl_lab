"""Detailed Eagle-inspired LunarLander skin."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Literal
from typing import TypeAlias

from gymnasium.envs.box2d import lunar_lander

from hpo.evaluation.rendering.solar_system_lander._skins._eagle_body_pygame import EAGLE_BODY_OPS
from hpo.evaluation.rendering.solar_system_lander._skins._eagle_legs_pygame import EAGLE_LEG_OPS

Point: TypeAlias = tuple[float, float]
Op: TypeAlias = tuple[str, str | None, str | None, float, list[Point]]
RGB: TypeAlias = tuple[int, int, int]
HaloMode: TypeAlias = Literal["auto", "always", "never"]

_HALO_LUMA_THRESHOLD = 45.0
_HALO_OUTLINE_COLOR = (230, 235, 240)
_HALO_WIDTH_ADD = 2
_HALO_MIN_OUTLINE_AREA = 500.0
_LEG_ANCHOR_Y_OFFSET = 18.75


@dataclass(frozen=True)
class DetailedEagleSkin:
    """Draw the detailed Eagle body and Gym-coupled Eagle legs."""

    scale: float = 0.16
    body_anchor: Point = (244.0, 236.0)
    right_leg_rest_angle: float = 0.492
    left_leg_rest_angle: float = -0.492
    halo: HaloMode = "auto"

    def __post_init__(self) -> None:
        if self.halo not in ("auto", "always", "never"):
            raise ValueError(f"unknown halo mode: {self.halo}")
        left_ops, right_ops = _split_leg_ops(EAGLE_LEG_OPS)
        object.__setattr__(self, "_body_outline_ops", _outline_ops(EAGLE_BODY_OPS))
        object.__setattr__(self, "_left_leg_ops", left_ops)
        object.__setattr__(self, "_right_leg_ops", right_ops)
        object.__setattr__(self, "_left_leg_outline_ops", _outline_ops(left_ops))
        object.__setattr__(self, "_right_leg_outline_ops", _outline_ops(right_ops))
        object.__setattr__(self, "_left_leg_anchor", _lift_leg_anchor(_bbox_center(left_ops)))
        object.__setattr__(self, "_right_leg_anchor", _lift_leg_anchor(_bbox_center(right_ops)))

    def draw(self, surface, env) -> None:
        """Draw the skin on an already screen-oriented LunarLander surface."""
        if getattr(env, "lander", None) is None or len(getattr(env, "legs", ())) < 2:
            return

        if _should_draw_halo(surface, env.lander.position, self.halo):
            self._draw(surface, env, outline_only=True)
        self._draw(surface, env, outline_only=False)

    def _draw(self, surface, env, *, outline_only: bool) -> None:
        body_ops = self._body_outline_ops if outline_only else EAGLE_BODY_OPS
        left_leg_ops = self._left_leg_outline_ops if outline_only else self._left_leg_ops
        right_leg_ops = self._right_leg_outline_ops if outline_only else self._right_leg_ops
        _draw_ops(
            surface, body_ops, env.lander, self.body_anchor, scale=self.scale, outline_only=outline_only
        )
        _draw_ops(
            surface,
            left_leg_ops,
            env.legs[1],
            self._left_leg_anchor,
            scale=self.scale,
            angle_offset=self.left_leg_rest_angle,
            outline_only=outline_only,
        )
        _draw_ops(
            surface,
            right_leg_ops,
            env.legs[0],
            self._right_leg_anchor,
            scale=self.scale,
            angle_offset=self.right_leg_rest_angle,
            outline_only=outline_only,
        )


def _draw_ops(
    surface,
    ops: Iterable[Op],
    body,
    source_anchor: Point,
    *,
    scale: float,
    angle_offset: float = 0.0,
    outline_only: bool = False,
) -> None:
    import pygame

    for kind, fill_hex, stroke_hex, stroke_width, points in ops:
        op = (kind, fill_hex, stroke_hex, stroke_width, points)
        if _draw_lod_op(surface, op, body, source_anchor, scale, angle_offset, outline_only):
            continue
        transformed = [_source_to_screen(point, body, source_anchor, scale, angle_offset) for point in points]
        fill = _rgb(fill_hex)
        stroke = _rgb(stroke_hex)
        if outline_only:
            stroke = _HALO_OUTLINE_COLOR
            fill = None
            width = max(1, round(stroke_width * scale)) + _HALO_WIDTH_ADD
        else:
            width = max(1, round(stroke_width * scale))

        if kind == "polygon":
            if fill is not None:
                pygame.draw.polygon(surface, fill, transformed, 0)
            if stroke is not None and len(transformed) >= 2:
                pygame.draw.polygon(surface, stroke, transformed, width)
        elif kind == "polyline":
            color = stroke or fill
            if color is not None and len(transformed) >= 2:
                pygame.draw.lines(surface, color, False, transformed, width)


def _draw_lod_op(
    surface, op: Op, body, source_anchor: Point, scale: float, angle_offset: float, outline_only: bool
) -> bool:
    return _draw_tiny_circle_op(surface, op, body, source_anchor, scale, angle_offset, outline_only)


def _draw_tiny_circle_op(
    surface, op: Op, body, source_anchor: Point, scale: float, angle_offset: float, outline_only: bool
) -> bool:
    kind, fill_hex, stroke_hex, stroke_width, points = op
    if outline_only or not _is_tiny_circle_op(op, scale):
        return False

    from pygame import gfxdraw

    min_x, min_y, max_x, max_y = _bbox(points)
    center = _source_to_screen(
        ((min_x + max_x) / 2, (min_y + max_y) / 2), body, source_anchor, scale, angle_offset
    )
    radius = max(1, round(((max_x - min_x) + (max_y - min_y)) * scale / 4))
    fill = _rgb(fill_hex)
    stroke = _rgb(stroke_hex)

    if fill is not None:
        gfxdraw.filled_circle(surface, center[0], center[1], radius, fill)
        gfxdraw.aacircle(surface, center[0], center[1], radius, fill)
    if stroke is not None:
        width = max(1, round(stroke_width * scale))
        for offset in range(width):
            gfxdraw.aacircle(surface, center[0], center[1], max(1, radius - offset), stroke)
    return True


def _is_tiny_circle_op(op: Op, scale: float) -> bool:
    kind, _fill_hex, stroke_hex, _stroke_width, points = op
    if kind != "polygon" or len(points) < 18:
        return False
    if stroke_hex is None:
        return False
    min_x, min_y, max_x, max_y = _bbox(points)
    width = max_x - min_x
    height = max_y - min_y
    if min(width, height) <= 0 or abs(width - height) > 2.0:
        return False
    return max(width, height) * scale <= 8.0


def _source_to_screen(
    point: Point, body, source_anchor: Point, scale: float, angle_offset: float
) -> tuple[int, int]:
    dx = (point[0] - source_anchor[0]) * scale
    dy = (point[1] - source_anchor[1]) * scale
    angle = -(float(body.angle) - angle_offset)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    origin = _body_to_screen(body.position)
    return (round(origin[0] + dx * cos_a - dy * sin_a), round(origin[1] + dx * sin_a + dy * cos_a))


def _body_to_screen(position) -> tuple[int, int]:
    return round(position[0] * lunar_lander.SCALE), round(
        lunar_lander.VIEWPORT_H - position[1] * lunar_lander.SCALE
    )


def _should_draw_halo(surface, lander_position, halo: HaloMode) -> bool:
    if halo == "always":
        return True
    if halo == "never":
        return False
    return _mean_luma_around(surface, _body_to_screen(lander_position)) < _HALO_LUMA_THRESHOLD


def _mean_luma_around(surface, center: tuple[int, int]) -> float:
    import pygame

    radius = 36
    rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    rect.clamp_ip(surface.get_rect())
    if rect.width <= 0 or rect.height <= 0:
        return 255.0
    view = pygame.surfarray.pixels3d(surface.subsurface(rect))
    try:
        return float((0.2126 * view[:, :, 0] + 0.7152 * view[:, :, 1] + 0.0722 * view[:, :, 2]).mean())
    finally:
        del view


def _split_leg_ops(ops: Sequence[Op]) -> tuple[list[Op], list[Op]]:
    left = []
    right = []
    for op in ops:
        center_x = _bbox_center([op])[0]
        if center_x < 245:
            left.append(op)
        else:
            right.append(op)
    return left, right


def _outline_ops(ops: Sequence[Op]) -> list[Op]:
    return [op for op in ops if _bbox_area(op) >= _HALO_MIN_OUTLINE_AREA]


def _bbox_area(op: Op) -> float:
    min_x, min_y, max_x, max_y = _bbox(op[4])
    return (max_x - min_x) * (max_y - min_y)


def _bbox_center(ops: Sequence[Op]) -> Point:
    points = [point for *_rest, op_points in ops for point in op_points]
    min_x, min_y, max_x, max_y = _bbox(points)
    return (min_x + max_x) / 2, (min_y + max_y) / 2


def _bbox(points: Sequence[Point]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _lift_leg_anchor(anchor: Point) -> Point:
    return anchor[0], anchor[1] + _LEG_ANCHOR_Y_OFFSET


def _rgb(hex_color: str | None) -> RGB | None:
    if hex_color is None or hex_color == "none":
        return None
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))
