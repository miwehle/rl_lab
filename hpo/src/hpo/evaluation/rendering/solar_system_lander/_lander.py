"""Lander drawing and optional skin protocol."""

from typing import Protocol

from gymnasium.envs.box2d import lunar_lander

from hpo.evaluation.rendering.solar_system_lander._colors import LanderColors, RGB


class LanderSkin(Protocol):
    """Optional visual replacement for the default Gym lander body."""

    def draw(self, surface, env, *, render_scale: int = 1) -> None:
        """Draw the lander on an already screen-oriented surface."""


def draw_gym_objects(
    surface, env, colors: LanderColors, gfxdraw, *, hide_lander_body: bool, render_scale: int = 1
) -> None:
    scale = lunar_lander.SCALE * render_scale
    for obj in env.particles + env.drawlist:
        if hide_lander_body and is_lander_body(obj, env):
            continue
        for fixture in obj.fixtures:
            trans = fixture.body.transform
            fill, outline = _object_colors(obj, env, colors)
            if type(fixture.shape) is lunar_lander.circleShape:
                center = trans * fixture.shape.pos * scale
                radius = fixture.shape.radius * scale
                _draw_circle(surface, center, radius, fill, outline)
            else:
                path = [trans * vertex * scale for vertex in fixture.shape.vertices]
                _draw_polygon(surface, gfxdraw, path, fill, outline)


def is_lander_body(obj, env) -> bool:
    return obj is env.lander or any(obj is leg for leg in env.legs)


def _object_colors(obj, env, colors: LanderColors) -> tuple[RGB, RGB]:
    if is_lander_body(obj, env):
        return colors.lander_fill, colors.lander_outline
    return obj.color1, obj.color2


def _draw_circle(surface, center, radius: float, fill: RGB, outline: RGB) -> None:
    import pygame

    pygame.draw.circle(surface, color=fill, center=center, radius=radius)
    pygame.draw.circle(surface, color=outline, center=center, radius=radius)


def _draw_polygon(surface, gfxdraw, path, fill: RGB, outline: RGB) -> None:
    import pygame

    pygame.draw.polygon(surface, color=fill, points=path)
    gfxdraw.aapolygon(surface, path, fill)
    pygame.draw.aalines(surface, color=outline, points=path, closed=True)
