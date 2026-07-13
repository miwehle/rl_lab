"""Colors and color scales for SolarSystemLander rendering."""

from collections.abc import Iterable
from dataclasses import dataclass

RGB = tuple[int, int, int]

WIND_COLOR_STOPS = (
    (0.0, (72, 220, 92)),
    (1.0, (255, 230, 72)),
    (2.0, (255, 146, 43)),
    (3.0, (255, 55, 55)),
)
TURBULENCE_COLOR_STOPS = (
    (0.0, (72, 220, 92)),
    (0.7, (255, 230, 72)),
    (1.4, (255, 146, 43)),
    (2.1, (255, 55, 55)),
)
KICK_COLOR_STOPS = (
    (0.0, (72, 220, 92)),
    (1.5, (255, 230, 72)),
    (3.0, (255, 146, 43)),
    (4.5, (255, 55, 55)),
)


@dataclass(frozen=True)
class LanderColors:
    sky: RGB = (0, 0, 0)
    ground: RGB = (255, 255, 255)
    ground_outline: RGB = (0, 0, 0)
    lander_fill: RGB = (128, 102, 230)
    lander_outline: RGB = (77, 77, 128)
    flag_pole: RGB = (255, 255, 255)
    flag: RGB = (204, 204, 0)


@dataclass(frozen=True)
class LanderOverlay:
    text_color: RGB = (255, 255, 255)
    shadow_color: RGB = (0, 0, 0)


_WORLD_COLORS = {
    "earth": LanderColors(sky=(143, 199, 232), ground=(111, 127, 82), ground_outline=(111, 127, 82)),
    "venus": LanderColors(sky=(214, 168, 92), ground=(138, 103, 64), ground_outline=(138, 103, 64)),
    "mars": LanderColors(sky=(201, 149, 122), ground=(139, 79, 61), ground_outline=(139, 79, 61)),
    "moon": LanderColors(sky=(5, 7, 10), ground=(119, 122, 125), ground_outline=(119, 122, 125)),
    "mercury": LanderColors(sky=(5, 7, 10), ground=(111, 107, 100), ground_outline=(111, 107, 100)),
}


def world_colors(worlds: Iterable[str]) -> list[LanderColors]:
    """Return render colors in the same order as the requested worlds."""
    colors = []
    for world in worlds:
        name = str(world)
        try:
            colors.append(_WORLD_COLORS[name])
        except KeyError:
            raise ValueError(f"unknown world color: {name}") from None
    return colors


def _force_color(value: float, stops: tuple[tuple[float, RGB], ...]) -> RGB:
    if value <= stops[0][0]:
        return stops[0][1]
    for (lower_value, lower_color), (upper_value, upper_color) in zip(stops, stops[1:]):
        if value <= upper_value:
            ratio = (value - lower_value) / (upper_value - lower_value)
            return _interpolate_color(lower_color, upper_color, ratio)
    return stops[-1][1]


def _interpolate_color(lower: RGB, upper: RGB, ratio: float) -> RGB:
    return tuple(round(low + (high - low) * ratio) for low, high in zip(lower, upper))
