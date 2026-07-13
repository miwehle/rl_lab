"""Small render presets for SolarSystemLander videos."""

from collections.abc import Iterable
from dataclasses import dataclass

from hpo.evaluation.rendering.solar_system_lander.colors import LanderColors, LanderOverlay, world_colors
from hpo.evaluation.rendering.solar_system_lander.lander import LanderSkin
from hpo.evaluation.rendering.solar_system_lander.skins import DetailedEagleSkin


@dataclass(frozen=True)
class RenderConfig:
    colors_by_world: tuple[LanderColors | None, ...]
    overlay: LanderOverlay | None = None
    skin: LanderSkin | None = None


def render_config(
    worlds: Iterable[str],
    *,
    overlay: bool = False,
    skin: str | None = None,
) -> RenderConfig:
    """Build the usual SolarSystemLander video render configuration."""
    render_skin: LanderSkin | None
    if skin is None:
        render_skin = None
    elif skin == "detailed_eagle":
        render_skin = DetailedEagleSkin()
    else:
        raise ValueError(f"unknown render skin: {skin}")

    return RenderConfig(
        colors_by_world=tuple(world_colors(worlds)),
        overlay=LanderOverlay() if overlay else None,
        skin=render_skin,
    )
