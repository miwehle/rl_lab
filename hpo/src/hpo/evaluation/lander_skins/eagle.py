"""Detailed Eagle-inspired LunarLander skin."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import TypeAlias

from gymnasium.envs.box2d import lunar_lander

from hpo.evaluation.lander_skins._eagle_body_pygame import EAGLE_BODY_OPS
from hpo.evaluation.lander_skins._eagle_legs_pygame import EAGLE_LEG_OPS

Point: TypeAlias = tuple[float, float]
Op: TypeAlias = tuple[str, str | None, str | None, float, list[Point]]
RGB: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class DetailedEagleSkin:
    """Draw the detailed Eagle body and Gym-coupled Eagle legs."""

    scale: float = 0.16
    body_anchor: Point = (244.0, 236.0)
    right_leg_rest_angle: float = 0.492
    left_leg_rest_angle: float = -0.492

    def __post_init__(self) -> None:
        left_ops, right_ops = _split_leg_ops(EAGLE_LEG_OPS)
        object.__setattr__(self, "_left_leg_ops", left_ops)
        object.__setattr__(self, "_right_leg_ops", right_ops)
        object.__setattr__(self, "_left_leg_anchor", _bbox_center(left_ops))
        object.__setattr__(self, "_right_leg_anchor", _bbox_center(right_ops))

    def draw(self, surface, env) -> None:
        """Draw the skin on an already screen-oriented LunarLander surface."""
        if getattr(env, "lander", None) is None or len(getattr(env, "legs", ())) < 2:
            return

        _draw_ops(surface, EAGLE_BODY_OPS, env.lander, self.body_anchor, scale=self.scale)
        _draw_ops(
            surface,
            self._left_leg_ops,
            env.legs[1],
            self._left_leg_anchor,
            scale=self.scale,
            angle_offset=self.left_leg_rest_angle,
        )
        _draw_ops(
            surface,
            self._right_leg_ops,
            env.legs[0],
            self._right_leg_anchor,
            scale=self.scale,
            angle_offset=self.right_leg_rest_angle,
        )


def _draw_ops(
    surface,
    ops: Iterable[Op],
    body,
    source_anchor: Point,
    *,
    scale: float,
    angle_offset: float = 0.0,
) -> None:
    import pygame

    for kind, fill_hex, stroke_hex, stroke_width, points in ops:
        transformed = [_source_to_screen(point, body, source_anchor, scale, angle_offset) for point in points]
        fill = _rgb(fill_hex)
        stroke = _rgb(stroke_hex)
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


def _source_to_screen(
    point: Point,
    body,
    source_anchor: Point,
    scale: float,
    angle_offset: float,
) -> tuple[int, int]:
    dx = (point[0] - source_anchor[0]) * scale
    dy = (point[1] - source_anchor[1]) * scale
    angle = -(float(body.angle) - angle_offset)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    origin = _body_to_screen(body.position)
    return (
        round(origin[0] + dx * cos_a - dy * sin_a),
        round(origin[1] + dx * sin_a + dy * cos_a),
    )


def _body_to_screen(position) -> tuple[int, int]:
    return round(position[0] * lunar_lander.SCALE), round(lunar_lander.VIEWPORT_H - position[1] * lunar_lander.SCALE)


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


def _bbox_center(ops: Sequence[Op]) -> Point:
    points = [point for *_rest, op_points in ops for point in op_points]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _rgb(hex_color: str | None) -> RGB | None:
    if hex_color is None or hex_color == "none":
        return None
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))
