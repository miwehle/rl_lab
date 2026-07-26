"""Internal PyVista rendering for 3D NN state snapshots."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import nn_viz.color_scheme as color_scheme
from nn_viz._rendering import (
    NetworkState,
    input_scale,
    node_fallback_scales,
    node_value,
    scale_value,
    source_value,
)
from nn_viz.layout import Edge
from nn_viz.layout import NetworkLayout, Node

_BACKGROUND = "white"
_EDGE_GEOMETRY_DEFAULT = "tube"
_NODE_RADIUS_DEFAULT = 0.055
_MIN_TUBE_RADIUS = 0.006
_MAX_TUBE_RADIUS = 0.016
_EDGE_LOW_INTENSITY_COLOR = (210, 210, 210)


def render_state_snapshot(
    layout: NetworkLayout,
    state: NetworkState,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    node_radius: float = _NODE_RADIUS_DEFAULT,
    edge_geometry: str = _EDGE_GEOMETRY_DEFAULT,
    scales: Mapping[str, Any] | None = None,
    edge_intensity: str = "saturation",
) -> Path:
    """Render an offscreen 3D layout snapshot for one NN state."""
    plotter = _state_plotter(
        layout,
        state,
        width=width,
        height=height,
        node_radius=node_radius,
        edge_geometry=edge_geometry,
        scales=scales,
        edge_intensity=edge_intensity,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter.screenshot(filename=str(output_path))
    finally:
        plotter.close()
    return output_path


def render_state_html(
    layout: NetworkLayout,
    state: NetworkState,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    node_radius: float = _NODE_RADIUS_DEFAULT,
    edge_geometry: str = _EDGE_GEOMETRY_DEFAULT,
    scales: Mapping[str, Any] | None = None,
    edge_intensity: str = "saturation",
) -> Path:
    """Render an interactive PyVista 3D scene for one NN state to HTML."""
    plotter = _state_plotter(
        layout,
        state,
        width=width,
        height=height,
        node_radius=node_radius,
        edge_geometry=edge_geometry,
        scales=scales,
        edge_intensity=edge_intensity,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plotter.export_html(str(output_path))
    finally:
        plotter.close()
    return output_path


def _state_plotter(
    layout: NetworkLayout,
    state: NetworkState,
    *,
    width: int,
    height: int,
    node_radius: float,
    edge_geometry: str,
    scales: Mapping[str, Any] | None,
    edge_intensity: str,
):
    if edge_intensity not in {"saturation", "opacity"}:
        raise ValueError("edge_intensity must be 'saturation' or 'opacity'")
    pv = load_pyvista()
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background(_BACKGROUND)
    _add_state_edges(plotter, pv, layout, state, edge_geometry, scales, edge_intensity)
    _add_state_nodes(plotter, pv, layout.nodes, state, node_radius, scales)
    _set_semantic_camera(plotter, layout.nodes)
    plotter.show_axes()
    return plotter


def _add_state_nodes(
    plotter,
    pv,
    nodes: tuple[Node, ...],
    state: NetworkState,
    radius: float,
    scales: Mapping[str, Any] | None,
) -> None:
    fallback_scales = node_fallback_scales(state)
    for node in nodes:
        sphere = pv.Sphere(radius=radius, center=(node.x, node.y, node.z))
        plotter.add_mesh(
            sphere,
            color=_rgb_hex(_state_node_color(node, state, scales, fallback_scales)),
            smooth_shading=True,
        )


def _add_state_edges(
    plotter,
    pv,
    layout: NetworkLayout,
    state: NetworkState,
    edge_geometry: str,
    scales: Mapping[str, Any] | None,
    edge_intensity: str,
) -> None:
    if edge_geometry not in {"line", "tube"}:
        raise ValueError("edge_geometry must be 'line' or 'tube'")
    nodes = {(node.layer, node.index): node for node in layout.nodes}
    weight_scale = _scale_value(scales, "weight", _weight_scale(layout.edges))
    activation_scale = _scale_value(scales, "activation", _max_source_magnitude(layout.edges, state))
    for edge in layout.edges:
        source = nodes.get((edge.source_layer, edge.source_index))
        target = nodes.get((edge.target_layer, edge.target_index))
        if source is None or target is None:
            continue
        line = pv.Line((source.x, source.y, source.z), (target.x, target.y, target.z))
        source_activation = source_value(edge, state)
        contribution = source_activation * edge.weight
        color_value = np.copysign(abs(edge.weight), contribution)
        color = _rgb_hex(_edge_color(color_value, weight_scale, source_activation, activation_scale, edge_intensity))
        opacity = _edge_opacity(source_activation, activation_scale, edge_intensity)
        if edge_geometry == "line":
            line_width = max(1, int(round(1.0 + color_scheme.edge_width(edge.weight, weight_scale))))
            plotter.add_mesh(line, color=color, line_width=line_width, opacity=opacity)
        else:
            tube = line.tube(radius=_tube_radius(edge.weight, weight_scale), n_sides=8)
            plotter.add_mesh(tube, color=color, opacity=opacity, smooth_shading=True)


def _weight_scale(edges: tuple[Edge, ...]) -> float:
    weights = np.asarray([abs(edge.weight) for edge in edges], dtype=np.float32)
    return float(np.percentile(weights, 95)) if weights.size else 0.0


def _tube_radius(weight: float, weight_scale: float) -> float:
    ratio = color_scheme.edge_width(weight, weight_scale) / 2.0
    return _MIN_TUBE_RADIUS + ratio * (_MAX_TUBE_RADIUS - _MIN_TUBE_RADIUS)


def _state_node_color(
    node: Node, state: NetworkState, scales: Mapping[str, Any] | None, fallback_scales: dict[str, float]
) -> tuple[int, int, int]:
    value = node_value(node, state)
    if node.layer == "in":
        scale = input_scale(scales, node.index, fallback_scales["input"])
        return color_scheme.signed_color(value, scale)
    if node.layer in {"h1", "h2"}:
        scale = _scale_value(scales, "hidden", fallback_scales["hidden"])
        return color_scheme.signed_color(value, scale)
    if node.layer == "out":
        scale = _scale_value(scales, "output", fallback_scales["output"])
        return color_scheme.signed_color(value, scale)
    return (128, 128, 128)


def _scale_value(scales: Mapping[str, Any] | None, key: str, fallback: float) -> float:
    return scale_value(scales, key, fallback)


def _max_source_magnitude(edges: tuple[Edge, ...], state: NetworkState) -> float:
    return max((abs(source_value(edge, state)) for edge in edges), default=0.0)


def _edge_opacity(source_activation: float, activation_scale: float, edge_intensity: str) -> float:
    if edge_intensity == "opacity":
        return color_scheme.alpha(source_activation, activation_scale) / 255.0
    return 1.0


def _edge_color(
    color_value: float,
    weight_scale: float,
    source_activation: float,
    activation_scale: float,
    edge_intensity: str,
) -> tuple[int, int, int]:
    base_color = color_scheme.signed_color(color_value, weight_scale)
    if edge_intensity == "saturation":
        ratio = color_scheme.alpha(source_activation, activation_scale) / 255.0
        return _mix_rgb(_EDGE_LOW_INTENSITY_COLOR, base_color, ratio)
    return base_color


def _set_semantic_camera(plotter, nodes: tuple[Node, ...]) -> None:
    center, span = _scene_center_and_span(nodes)
    x, y, z = center
    plotter.enable_parallel_projection()
    plotter.camera_position = [
        (x, y - 2.0 * span, z + 0.9 * span),
        center,
        (0.0, 0.0, 1.0),
    ]
    if hasattr(plotter, "camera"):
        plotter.camera.parallel_scale = 0.65 * span
    if hasattr(plotter, "reset_camera_clipping_range"):
        plotter.reset_camera_clipping_range()


def _scene_center_and_span(nodes: tuple[Node, ...]) -> tuple[tuple[float, float, float], float]:
    if not nodes:
        return (0.0, 0.0, 0.0), 1.0
    points = np.asarray([(node.x, node.y, node.z) for node in nodes], dtype=np.float32)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = tuple(float(value) for value in (lower + upper) / 2.0)
    span = float(np.max(upper - lower))
    return center, max(span, 1.0)


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _mix_rgb(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    mixed = np.asarray(start, dtype=np.float32) * (1.0 - ratio) + np.asarray(end, dtype=np.float32) * ratio
    return tuple(int(round(value)) for value in np.clip(mixed, 0, 255))


@lru_cache(maxsize=1)
def load_pyvista():
    try:
        import pyvista
    except ImportError as exc:
        raise RuntimeError("3D snapshots require the pyvista package") from exc
    return pyvista
