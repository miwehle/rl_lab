"""Internal PyVista rendering for 3D NN layout snapshots."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import nn_viz.color_scheme as color_scheme
from nn_viz.layout import Edge
from nn_viz.layout import NetworkLayout, Node

_BACKGROUND = "white"
_EDGE_GEOMETRY_DEFAULT = "tube"
_NEUTRAL_EDGE_COLOR = "#9ca3af"
_NODE_RADIUS_DEFAULT = 0.055
_MIN_TUBE_RADIUS = 0.006
_MAX_TUBE_RADIUS = 0.026


def render_layout_snapshot(
    layout: NetworkLayout,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    node_radius: float = _NODE_RADIUS_DEFAULT,
    edge_geometry: str = _EDGE_GEOMETRY_DEFAULT,
) -> Path:
    """Render an offscreen 3D layout snapshot to output_path."""
    pv = load_pyvista()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    try:
        plotter.set_background(_BACKGROUND)
        _add_edges(plotter, pv, layout, edge_geometry)
        _add_nodes(plotter, pv, layout.nodes, node_radius, _activity_scale(layout.nodes))
        plotter.enable_parallel_projection()
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.screenshot(filename=str(output_path))
    finally:
        plotter.close()
    return output_path


def _add_nodes(plotter, pv, nodes: tuple[Node, ...], radius: float, activity_scale: float) -> None:
    for node in nodes:
        sphere = pv.Sphere(radius=radius, center=(node.x, node.y, node.z))
        plotter.add_mesh(
            sphere,
            color=_rgb_hex(color_scheme.heat_color(node.activity, activity_scale)),
            smooth_shading=True,
        )


def _add_edges(plotter, pv, layout: NetworkLayout, edge_geometry: str) -> None:
    if edge_geometry not in {"line", "tube"}:
        raise ValueError("edge_geometry must be 'line' or 'tube'")
    nodes = {(node.layer, node.index): node for node in layout.nodes}
    weight_scale = _weight_scale(layout.edges)
    for edge in layout.edges:
        source = nodes.get((edge.source_layer, edge.source_index))
        target = nodes.get((edge.target_layer, edge.target_index))
        if source is None or target is None:
            continue
        line = pv.Line((source.x, source.y, source.z), (target.x, target.y, target.z))
        color = _rgb_hex(color_scheme.signed_color(edge.weight, weight_scale))
        if edge_geometry == "line":
            line_width = max(1, int(round(color_scheme.edge_width(edge.weight, weight_scale))))
            plotter.add_mesh(line, color=color, line_width=line_width)
        else:
            tube = line.tube(radius=_tube_radius(edge.weight, weight_scale), n_sides=8)
            plotter.add_mesh(tube, color=color, smooth_shading=True)


def _activity_scale(nodes: tuple[Node, ...]) -> float:
    return max((node.activity for node in nodes), default=0.0)


def _weight_scale(edges: tuple[Edge, ...]) -> float:
    weights = np.asarray([abs(edge.weight) for edge in edges], dtype=np.float32)
    return float(np.percentile(weights, 95)) if weights.size else 0.0


def _tube_radius(weight: float, weight_scale: float) -> float:
    ratio = (color_scheme.edge_width(weight, weight_scale) - 1.0) / 2.0
    return _MIN_TUBE_RADIUS + ratio * (_MAX_TUBE_RADIUS - _MIN_TUBE_RADIUS)


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


@lru_cache(maxsize=1)
def load_pyvista():
    try:
        import pyvista
    except ImportError as exc:
        raise RuntimeError("3D snapshots require the pyvista package") from exc
    return pyvista
