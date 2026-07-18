"""Experimental colored Eagle skin with Gym-coupled leg sprites."""

from __future__ import annotations

from dataclasses import dataclass

from hpo.evaluation.rendering.solar_system_lander._skins.colored_eagle import (
    _ASSET_DIR,
    _Asset,
    _blit_on_body,
)

_LEFT_LEG_PNG = _ASSET_DIR / "eagle_colored_left_leg.png"
_RIGHT_LEG_PNG = _ASSET_DIR / "eagle_colored_right_leg.png"
_BODY_PNG = _ASSET_DIR / "eagle_colored_body.png"


@dataclass(frozen=True)
class ColoredEagle2Skin:
    """Draw colored Eagle sprites with legs coupled to Gym leg bodies."""

    scale: float = 0.16
    body_anchor: tuple[float, float] = (251.0, 332.0)
    left_leg_anchor: tuple[float, float] = (87.125, 128.875)
    right_leg_anchor: tuple[float, float] = (55.375, 128.875)
    right_leg_rest_angle: float = 0.344
    left_leg_rest_angle: float = -0.344

    def draw(self, surface, env, *, render_scale: int = 1) -> None:
        if getattr(env, "lander", None) is None or len(getattr(env, "legs", ())) < 2:
            return

        body, left_leg, right_leg = _assets()
        scale = self.scale * render_scale
        _blit_on_body(surface, left_leg.surface, env.legs[1], self.left_leg_anchor, scale, self.left_leg_rest_angle, render_scale=render_scale)
        _blit_on_body(surface, right_leg.surface, env.legs[0], self.right_leg_anchor, scale, self.right_leg_rest_angle, render_scale=render_scale)
        _blit_on_body(surface, body.surface, env.lander, self.body_anchor, scale, render_scale=render_scale)


_CACHED_ASSETS: tuple[_Asset, _Asset, _Asset] | None = None


def _assets() -> tuple[_Asset, _Asset, _Asset]:
    global _CACHED_ASSETS
    if _CACHED_ASSETS is None:
        import pygame

        _CACHED_ASSETS = (
            _Asset(pygame.image.load(str(_BODY_PNG)), (251.0, 332.0)),
            _Asset(pygame.image.load(str(_LEFT_LEG_PNG)), (87.125, 128.875)),
            _Asset(pygame.image.load(str(_RIGHT_LEG_PNG)), (55.375, 128.875)),
        )
    return _CACHED_ASSETS
