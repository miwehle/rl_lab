"""Photorealistic Eagle skin for LunarLander-compatible rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hpo.evaluation.rendering.solar_system_lander._skins.colored_eagle import _Asset, _blit_on_body

_ASSET_DIR = Path(__file__).resolve().parents[1] / "_skin_assets" / "eagle_photorealistic"
_EAGLE_PNG = _ASSET_DIR / "eagle_photorealistic.png"


@dataclass(frozen=True)
class PhotorealisticEagleSkin:
    """Draw a freestanding photorealistic Eagle PNG as a single sprite."""

    scale: float = 0.08
    anchor: tuple[float, float] = (512.0, 678.0)

    def draw(self, surface, env, *, render_scale: int = 1) -> None:
        if getattr(env, "lander", None) is None:
            return

        eagle = _asset()
        _blit_on_body(surface, eagle.surface, env.lander, self.anchor, self.scale * render_scale, render_scale=render_scale)


_CACHED_ASSET: _Asset | None = None


def _asset() -> _Asset:
    global _CACHED_ASSET
    if _CACHED_ASSET is None:
        import pygame

        _CACHED_ASSET = _Asset(pygame.image.load(str(_EAGLE_PNG)), (512.0, 678.0))
    return _CACHED_ASSET
