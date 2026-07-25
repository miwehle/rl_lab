"""Internal PyVista rendering for 3D NN layout snapshots."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nn_viz.layout import NetworkLayout, Node

_BACKGROUND = "white"
_EDGE_COLOR = "#9ca3af"
_NODE_COLORS = {
    "in": "#6b7280",
    "h1": "#8a8f98",
    "h2": "#2b6cb0",
    "out": "#dd6b20",
}


def render_layout_snapshot(
    layout: NetworkLayout,
    output_path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    node_radius: float = 0.055,
) -> Path:
    """Render a simple offscreen 3D layout snapshot to output_path."""
    pv = load_pyvista()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    try:
        plotter.set_background(_BACKGROUND)
        _add_edges(plotter, pv, layout)
        _add_nodes(plotter, pv, layout.nodes, node_radius)
        plotter.enable_parallel_projection()
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.screenshot(filename=str(output_path))
    finally:
        plotter.close()
    return output_path


def _add_nodes(plotter, pv, nodes: tuple[Node, ...], radius: float) -> None:
    for node in nodes:
        sphere = pv.Sphere(radius=radius, center=(node.x, node.y, node.z))
        plotter.add_mesh(
            sphere,
            color=_NODE_COLORS.get(node.layer, "#6b7280"),
            smooth_shading=True,
        )


def _add_edges(plotter, pv, layout: NetworkLayout) -> None:
    nodes = {(node.layer, node.index): node for node in layout.nodes}
    for edge in layout.edges:
        source = nodes.get((edge.source_layer, edge.source_index))
        target = nodes.get((edge.target_layer, edge.target_index))
        if source is None or target is None:
            continue
        line = pv.Line((source.x, source.y, source.z), (target.x, target.y, target.z))
        plotter.add_mesh(line, color=_EDGE_COLOR, line_width=1)


@lru_cache(maxsize=1)
def load_pyvista():
    try:
        import pyvista
    except ImportError as exc:
        raise RuntimeError("3D snapshots require the pyvista package") from exc
    return pyvista
