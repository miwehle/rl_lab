"""Render standalone images from saved NN video traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

import nn_viz.color_scheme as color_scheme
from nn_viz._edges import (
    EDGE_EFFECT_QUANTILE_DEFAULT,
    network_edges_from_trace,
    scales_with_edge_weight_scale,
    select_edges_by_effect,
)
from nn_viz.layout import Edge, NetworkLayout, Node
from nn_viz._pyvista_rendering import render_state_html, render_state_snapshot
from nn_viz._rendering import (
    EDGE_RENDERER_DEFAULT,
    EdgeStyle,
    NetworkState,
    default_node_outline,
    input_scale,
    max_source_magnitude,
    node_fallback_scales,
    node_value,
    render_layout_rgba,
    render_state_layout_rgba,
    scale_value,
    source_value,
)


def _load_trace_state(trace_path: str | Path, *, step: int, window_steps: int = 1) -> NetworkState:
    """Load one raw or backward-window-mean NN state from a saved trace."""
    with np.load(trace_path) as trace:
        return _trace_state_from_arrays(trace, step=step, window_steps=window_steps)


def render_trace_step(
    trace_path: str | Path,
    layout: NetworkLayout,
    output_path: str | Path,
    *,
    step: int,
    window_steps: int = 1,
    width: int = 1280,
    height: int = 360,
    scales: Mapping[str, Any] | None = None,
    edge_effect_quantile: float = EDGE_EFFECT_QUANTILE_DEFAULT,
    edge_renderer: str = EDGE_RENDERER_DEFAULT,
    label_mode: str = "indices",
) -> Path:
    """Render one trace step and save the image to output_path."""
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(trace_path) as trace:
        state = _trace_state_from_arrays(trace, step=step, window_steps=window_steps)
        all_edges = network_edges_from_trace(trace, layout)
        render_scales = scales_with_edge_weight_scale(scales, all_edges)
    rgba = render_state_layout_rgba(
        NetworkLayout(layout.nodes, select_edges_by_effect(all_edges, state, edge_effect_quantile)),
        state,
        width=width,
        height=height,
        scales=render_scales,
        edge_renderer=edge_renderer,
        label_mode=label_mode,
    )
    Image.fromarray(rgba).save(output_path)
    return output_path


def render_trace_step_3d(
    trace_path: str | Path,
    layout: NetworkLayout,
    output_path: str | Path,
    *,
    step: int,
    window_steps: int = 1,
    width: int = 1280,
    height: int = 720,
    scales: Mapping[str, Any] | None = None,
    edge_geometry: str = "tube",
    edge_intensity: str = "saturation",
    edge_effect_quantile: float = EDGE_EFFECT_QUANTILE_DEFAULT,
) -> Path:
    """Render one trace step as a PyVista 3D screenshot."""
    with np.load(trace_path) as trace:
        state = _trace_state_from_arrays(trace, step=step, window_steps=window_steps)
        all_edges = network_edges_from_trace(trace, layout)
        if scales is None:
            scales = _trace_scales_from_arrays(trace, layout)
        render_layout = NetworkLayout(layout.nodes, select_edges_by_effect(all_edges, state, edge_effect_quantile))
        render_scales = scales_with_edge_weight_scale(scales, all_edges)
    return render_state_snapshot(
        render_layout,
        state,
        output_path,
        width=width,
        height=height,
        scales=render_scales,
        edge_geometry=edge_geometry,
        edge_intensity=edge_intensity,
    )


def render_trace_step_3d_html(
    trace_path: str | Path,
    layout: NetworkLayout,
    output_path: str | Path,
    *,
    step: int,
    window_steps: int = 1,
    width: int = 1280,
    height: int = 720,
    scales: Mapping[str, Any] | None = None,
    edge_geometry: str = "tube",
    edge_intensity: str = "saturation",
    edge_effect_quantile: float = EDGE_EFFECT_QUANTILE_DEFAULT,
) -> Path:
    """Render one trace step as an interactive PyVista HTML scene."""
    with np.load(trace_path) as trace:
        state = _trace_state_from_arrays(trace, step=step, window_steps=window_steps)
        all_edges = network_edges_from_trace(trace, layout)
        if scales is None:
            scales = _trace_scales_from_arrays(trace, layout)
        render_layout = NetworkLayout(layout.nodes, select_edges_by_effect(all_edges, state, edge_effect_quantile))
        render_scales = scales_with_edge_weight_scale(scales, all_edges)
    return render_state_html(
        render_layout,
        state,
        output_path,
        width=width,
        height=height,
        scales=render_scales,
        edge_geometry=edge_geometry,
        edge_intensity=edge_intensity,
    )


def render_trace_diff(
    trace_path: str | Path,
    layout: NetworkLayout,
    output_path: str | Path,
    *,
    from_step: int,
    to_step: int,
    from_window_steps: int = 1,
    to_window_steps: int = 1,
    width: int = 1280,
    height: int = 360,
) -> Path:
    """Render to-window minus from-window differences and save the image to output_path."""
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(trace_path) as trace:
        from_state = _trace_state_from_arrays(trace, step=from_step, window_steps=from_window_steps)
        to_state = _trace_state_from_arrays(trace, step=to_step, window_steps=to_window_steps)
        scales = _trace_scales_from_arrays(trace, layout)
    diff_state = _diff_state(from_state, to_state)
    rgba = _render_diff_layout_rgba(layout, diff_state, scales=scales, width=width, height=height)
    Image.fromarray(rgba).save(output_path)
    return output_path


def _trace_state_from_arrays(
    trace: Mapping[str, np.ndarray], *, step: int, window_steps: int
) -> NetworkState:
    if window_steps < 1:
        raise ValueError("window_steps must be >= 1")
    steps = np.asarray(trace["steps"])
    matches = np.flatnonzero(steps == step)
    if matches.size == 0:
        raise ValueError(f"step {step} not found in trace")
    row_index = int(matches[0])
    start = max(0, row_index - window_steps + 1)
    stop = row_index + 1
    return NetworkState(
        inputs=np.mean(trace["observations"][start:stop], axis=0, dtype=np.float32),
        h1=np.mean(trace["h1"][start:stop], axis=0, dtype=np.float32),
        h2=np.mean(trace["h2"][start:stop], axis=0, dtype=np.float32),
        q_values=np.mean(trace["q_values"][start:stop], axis=0, dtype=np.float32),
        action=int(trace["actions"][row_index]),
    )


def _diff_state(from_state: NetworkState, to_state: NetworkState) -> NetworkState:
    return NetworkState(
        inputs=to_state.inputs - from_state.inputs,
        h1=to_state.h1 - from_state.h1,
        h2=to_state.h2 - from_state.h2,
        q_values=to_state.q_values - from_state.q_values,
        action=-1,
    )


def _trace_scales_from_arrays(trace: Mapping[str, np.ndarray], layout: NetworkLayout) -> dict[str, Any]:
    input_abs = np.abs(trace["observations"])
    hidden_values = np.concatenate([trace["h1"].ravel(), trace["h2"].ravel()])
    output_abs = np.abs(trace["q_values"])
    all_edges = network_edges_from_trace(trace, layout)
    weights = np.asarray([abs(edge.weight) for edge in all_edges], dtype=np.float32)
    return {
        "input": np.percentile(input_abs, 95, axis=0).astype(float),
        "hidden": float(np.percentile(hidden_values, 95)),
        "output": float(np.percentile(output_abs, 95)),
        "activation": float(np.percentile(np.concatenate([input_abs.ravel(), hidden_values]), 95)),
        "weight": float(np.percentile(weights, 95)) if weights.size else 1.0,
    }


def _render_diff_layout_rgba(
    layout: NetworkLayout, diff_state: NetworkState, *, scales: Mapping[str, Any], width: int, height: int
) -> np.ndarray:
    weight_scale = scale_value(
        scales, "weight", max((abs(edge.weight) for edge in layout.edges), default=0.0)
    )
    activation_scale = scale_value(scales, "activation", max_source_magnitude(layout.edges, diff_state))
    edge_scale = activation_scale * weight_scale
    fallback_scales = node_fallback_scales(diff_state)

    def node_fill(node: Node) -> tuple[int, int, int, int]:
        value = node_value(node, diff_state)
        if node.layer == "in":
            scale = input_scale(scales, node.index, fallback_scales["input"])
        elif node.layer in {"h1", "h2"}:
            scale = scale_value(scales, "hidden", fallback_scales["hidden"])
        elif node.layer == "out":
            scale = scale_value(scales, "output", fallback_scales["output"])
        else:
            scale = 0.0
        return (*color_scheme.signed_color(value, scale), color_scheme.alpha(value, scale))

    def edge_style(edge: Edge) -> EdgeStyle | None:
        edge_delta = source_value(edge, diff_state) * edge.weight
        return EdgeStyle(
            fill=(
                *color_scheme.signed_color(edge_delta, edge_scale),
                color_scheme.alpha(edge_delta, edge_scale),
            ),
            nominal_width=color_scheme.edge_width(edge.weight, weight_scale),
        )

    return render_layout_rgba(
        layout,
        width=width,
        height=height,
        node_fill=node_fill,
        node_outline=default_node_outline,
        edge_style=edge_style,
        edge_renderer="pillow",
        label_mode="indices",
    )
