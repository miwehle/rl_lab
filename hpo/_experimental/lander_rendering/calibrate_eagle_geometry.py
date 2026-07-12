"""Render one Eagle/Gymnasium geometry calibration frame.

This script is experimental. It checks whether detailed Eagle drawing ops can be
anchored to Gymnasium LunarLander physics without hiding important physical
signals such as leg motion, leg breakage, and nozzle locations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import argparse
import math
from pathlib import Path
import sys
from typing import TypeAlias

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.box2d import lunar_lander

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "hpo" / "src"
DQN_SRC_DIR = ROOT / "dqn" / "src"
DETAILED_DIR = Path(__file__).resolve().parent / "detailed"
sys.path.insert(0, str(DQN_SRC_DIR))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DETAILED_DIR))

from eagle_body_pygame import EAGLE_BODY_OPS  # noqa: E402
from eagle_legs_pygame import EAGLE_LEG_OPS  # noqa: E402
from hpo.evaluation.lander_rendering import LanderColors, LanderRenderWrapper  # noqa: E402

Point: TypeAlias = tuple[float, float]
Op: TypeAlias = tuple[str, str | None, str | None, float, list[Point]]
RGB: TypeAlias = tuple[int, int, int]

OUTPUT = Path(__file__).resolve().parent / "detailed" / "eagle_geometry_calibration.png"
EAGLE_SCALE = 0.16
BODY_ANCHOR = (244.0, 236.0)
MAIN_NOZZLE = (243.9, 358.0)
LEFT_SIDE_NOZZLE = (156.0, 140.0)
RIGHT_SIDE_NOZZLE = (332.0, 139.0)
RIGHT_LEG_REST_ANGLE = 0.492
LEFT_LEG_REST_ANGLE = -0.492

MARKER_BODY: RGB = (255, 45, 45)
MARKER_LEG: RGB = (34, 211, 238)
MARKER_FOOT: RGB = (80, 255, 80)
MARKER_MAIN: RGB = (255, 0, 255)
MARKER_SIDE: RGB = (255, 230, 0)
MARKER_TEXT: RGB = (0, 0, 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="earth", choices=["moon", "mercury", "mars", "earth", "venus"])
    parser.add_argument("--seed", type=int, default=10_014)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    pygame.init()
    env = _make_env(args.world)
    try:
        env.reset(seed=args.seed)
        for _ in range(args.steps):
            env.step(0)
        frame = env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        _draw_eagle(surface, env.unwrapped)
        zoom_surface = surface.copy()
        _draw_markers(zoom_surface, env.unwrapped, labels=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        zoom_output = args.output.with_name(args.output.stem + "_zoom" + args.output.suffix)
        _save_zoom(zoom_surface, env.unwrapped.lander.position, zoom_output)
        _draw_markers(surface, env.unwrapped, labels=True)
        pygame.image.save(surface, str(args.output))
        print(args.output)
        print(zoom_output)
    finally:
        env.close()
        pygame.quit()
    return 0


def _make_env(world_name: str):
    world = {
        "moon": (-1.65, (0.0, 0.0), (0.0, 0.0)),
        "mercury": (-3.7, (0.0, 0.0), (0.0, 0.0)),
        "mars": (-3.8, (0.0, 4.0), (0.0, 1.0)),
        "earth": (-10.0, (5.0, 15.0), (0.0, 2.0)),
        "venus": (-9.0, (15.0, 20.0), (0.0, 2.0)),
    }[world_name]
    gravity, wind_power, turbulence_power = world
    base = gym.make(
        "LunarLander-v3",
        gravity=gravity,
        enable_wind=wind_power[1] > 0 or turbulence_power[1] > 0,
        render_mode="rgb_array",
    )
    base.unwrapped.wind_power = wind_power[1]
    base.unwrapped.turbulence_power = turbulence_power[1]
    return LanderRenderWrapper(
        base,
        colors=LanderColors(sky=(143, 199, 232), ground=(111, 127, 82)),
        overlay=None,
    )


def _draw_eagle(surface: pygame.Surface, env) -> None:
    _draw_ops(surface, EAGLE_BODY_OPS, env.lander, BODY_ANCHOR)

    left_ops, right_ops = _split_leg_ops(EAGLE_LEG_OPS)
    _draw_ops(surface, left_ops, env.legs[1], _bbox_center(left_ops), angle_offset=LEFT_LEG_REST_ANGLE)
    _draw_ops(surface, right_ops, env.legs[0], _bbox_center(right_ops), angle_offset=RIGHT_LEG_REST_ANGLE)


def _draw_markers(surface: pygame.Surface, env, *, labels: bool) -> None:
    font = pygame.font.Font(None, 16)
    _mark(surface, _body_to_screen(env.lander.position), MARKER_BODY, "body", font, labels=labels)
    _mark(surface, _body_to_screen(env.legs[0].position), MARKER_LEG, "leg 0", font, labels=labels)
    _mark(surface, _body_to_screen(env.legs[1].position), MARKER_LEG, "leg 1", font, labels=labels)

    for index, leg in enumerate(env.legs):
        vertices = _fixture_vertices(leg)
        pygame.draw.lines(surface, MARKER_LEG, True, vertices, 1)
        foot = max(vertices, key=lambda point: point[1])
        _mark(surface, foot, MARKER_FOOT, f"foot {index}", font, labels=labels)

    _mark(surface, _main_engine_anchor(env.lander), MARKER_MAIN, "main", font, labels=labels)
    for direction, label in [(-1, "side L"), (1, "side R")]:
        _mark(surface, _side_engine_anchor(env.lander, direction), MARKER_SIDE, label, font, labels=labels)

    _mark(
        surface,
        _source_to_screen(MAIN_NOZZLE, env.lander, BODY_ANCHOR),
        MARKER_MAIN,
        "main art",
        font,
        labels=labels,
        radius=3,
    )
    _mark(
        surface,
        _source_to_screen(LEFT_SIDE_NOZZLE, env.lander, BODY_ANCHOR),
        MARKER_SIDE,
        "side art",
        font,
        labels=labels,
        radius=3,
    )
    _mark(
        surface,
        _source_to_screen(RIGHT_SIDE_NOZZLE, env.lander, BODY_ANCHOR),
        MARKER_SIDE,
        "side art",
        font,
        labels=labels,
        radius=3,
    )


def _draw_ops(
    surface: pygame.Surface,
    ops: Iterable[Op],
    body,
    source_anchor: Point,
    *,
    angle_offset: float = 0.0,
) -> None:
    for kind, fill_hex, stroke_hex, stroke_width, points in ops:
        transformed = [_source_to_screen(point, body, source_anchor, angle_offset) for point in points]
        fill = _rgb(fill_hex)
        stroke = _rgb(stroke_hex)
        width = max(1, round(stroke_width * EAGLE_SCALE))

        if kind == "polygon":
            if fill is not None:
                pygame.draw.polygon(surface, fill, transformed, 0)
            if stroke is not None and len(transformed) >= 2:
                pygame.draw.polygon(surface, stroke, transformed, width)
        elif kind == "polyline":
            color = stroke or fill
            if color is not None and len(transformed) >= 2:
                pygame.draw.lines(surface, color, False, transformed, width)


def _source_to_screen(point: Point, body, source_anchor: Point, angle_offset: float = 0.0) -> tuple[int, int]:
    dx = (point[0] - source_anchor[0]) * EAGLE_SCALE
    dy = (point[1] - source_anchor[1]) * EAGLE_SCALE
    angle = -(float(body.angle) - angle_offset)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    origin = _body_to_screen(body.position)
    return (
        round(origin[0] + dx * cos_a - dy * sin_a),
        round(origin[1] + dx * sin_a + dy * cos_a),
    )


def _main_engine_anchor(lander) -> tuple[int, int]:
    tip = (math.sin(lander.angle), math.cos(lander.angle))
    ox = tip[0] * lunar_lander.MAIN_ENGINE_Y_LOCATION / lunar_lander.SCALE
    oy = -tip[1] * lunar_lander.MAIN_ENGINE_Y_LOCATION / lunar_lander.SCALE
    return _body_to_screen((lander.position[0] + ox, lander.position[1] + oy))


def _side_engine_anchor(lander, direction: int) -> tuple[int, int]:
    tip = (math.sin(lander.angle), math.cos(lander.angle))
    side = (-tip[1], tip[0])
    ox = side[0] * direction * lunar_lander.SIDE_ENGINE_AWAY / lunar_lander.SCALE
    oy = -side[1] * direction * lunar_lander.SIDE_ENGINE_AWAY / lunar_lander.SCALE
    impulse = (
        lander.position[0] + ox - tip[0] * 17 / lunar_lander.SCALE,
        lander.position[1] + oy + tip[1] * lunar_lander.SIDE_ENGINE_HEIGHT / lunar_lander.SCALE,
    )
    return _body_to_screen(impulse)


def _fixture_vertices(body) -> list[tuple[int, int]]:
    fixture = body.fixtures[0]
    return [_body_to_screen(body.transform * vertex) for vertex in fixture.shape.vertices]


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


def _mark(
    surface: pygame.Surface,
    point: tuple[int, int],
    color: RGB,
    label: str,
    font: pygame.font.Font,
    *,
    labels: bool,
    radius: int = 4,
) -> None:
    pygame.draw.circle(surface, (0, 0, 0), (point[0] + 1, point[1] + 1), radius + 1)
    pygame.draw.circle(surface, color, point, radius)
    if not labels:
        return
    text = font.render(label, True, MARKER_TEXT)
    surface.blit(text, (point[0] + 6, point[1] - 6))


def _save_zoom(surface: pygame.Surface, center_position, output: Path) -> None:
    center = _body_to_screen(center_position)
    crop_size = 150
    rect = pygame.Rect(
        max(0, center[0] - crop_size // 2),
        max(0, center[1] - crop_size // 2),
        crop_size,
        crop_size,
    )
    rect.clamp_ip(surface.get_rect())
    crop = surface.subsurface(rect).copy()
    zoom = pygame.transform.scale(crop, (crop.get_width() * 4, crop.get_height() * 4))
    pygame.image.save(zoom, str(output))


def _rgb(hex_color: str | None) -> RGB | None:
    if hex_color is None or hex_color == "none":
        return None
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))


if __name__ == "__main__":
    raise SystemExit(main())
