"""Color and width mappings for live NN visualization."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np

RGB: TypeAlias = tuple[int, int, int]

_BLUE = (37, 99, 235)
_GRAY = (128, 128, 128)
_RED = (220, 38, 38)
_HEAT_STOPS: tuple[tuple[float, RGB], ...] = (
    (0.0, (31, 41, 55)),
    (0.35, (185, 28, 28)),
    (0.7, (245, 158, 11)),
    (1.0, (255, 255, 235)),
)


def alpha(value: float, scale: float) -> int:
    """Map value magnitude to an alpha channel."""
    return int(round(255 * _log_ratio(value, scale)))


def signed_color(value: float, scale: float) -> RGB:
    """Map signed values to blue/gray/red."""
    ratio = _linear_ratio(value, scale)
    if ratio == 0.0:
        return _GRAY
    target = _RED if value > 0.0 else _BLUE
    return _mix_rgb(_GRAY, target, ratio)


def heat_color(value: float, scale: float) -> RGB:
    """Map nonnegative activations to a heat-like color."""
    ratio = _log_ratio(max(value, 0.0), scale)
    for (lower_ratio, lower_color), (upper_ratio, upper_color) in zip(_HEAT_STOPS, _HEAT_STOPS[1:]):
        if ratio <= upper_ratio:
            local = (ratio - lower_ratio) / max(1e-9, upper_ratio - lower_ratio)
            return _mix_rgb(lower_color, upper_color, local)
    return _HEAT_STOPS[-1][1]


def edge_width(weight: float, scale: float) -> float:
    """Map weight magnitude to a nominal edge width."""
    return 1.0 + 2.0 * _log_ratio(weight, scale)


def _log_ratio(value: float, scale: float) -> float:
    scale = abs(float(scale))
    if scale <= 0.0:
        return 0.0
    ratio = math.log1p(abs(float(value))) / math.log1p(scale)
    return float(np.clip(ratio, 0.0, 1.0))


def _linear_ratio(value: float, scale: float) -> float:
    scale = abs(float(scale))
    if scale <= 0.0:
        return 0.0
    return float(np.clip(abs(float(value)) / scale, 0.0, 1.0))


def _mix_rgb(start: RGB, end: RGB, ratio: float) -> RGB:
    mixed = np.asarray(start, dtype=np.float32) * (1.0 - ratio) + np.asarray(end, dtype=np.float32) * ratio
    return tuple(int(round(value)) for value in np.clip(mixed, 0, 255))
