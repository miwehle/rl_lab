"""Render standalone images from saved NN video traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

import nn_viz.color_scheme as color_scheme
from nn_viz.layout import Edge, NetworkLayout, Node
from nn_viz.rendering import (
    _EDGE_RENDERER_DEFAULT,
    _EDGE_SKIP_ACTIVATION_DEFAULT,
    _EDGE_SKIP_WEIGHT_DEFAULT,
    _EdgeStyle,
    _NetworkState,
    _default_node_outline,
    _input_scale,
    _max_source_magnitude,
    _node_fallback_scales,
    _node_value,
    _render_layout_rgba,
    _render_state_layout_rgba,
    _scale_value,
    _source_value,
)


def _load_trace_state(trace_path: str | Path, *, step: int, window_steps: int = 1) -> _NetworkState:
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
    edge_skip_activation: float = _EDGE_SKIP_ACTIVATION_DEFAULT,
    edge_skip_weight: float = _EDGE_SKIP_WEIGHT_DEFAULT,
    edge_renderer: str = _EDGE_RENDERER_DEFAULT,
    label_mode: str = "indices",
) -> Path:
    """Render one trace step and save the image to output_path."""
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = _load_trace_state(trace_path, step=step, window_steps=window_steps)
    rgba = _render_state_layout_rgba(
        layout,
        state,
        width=width,
        height=height,
        scales=scales,
        edge_skip_activation=edge_skip_activation,
        edge_skip_weight=edge_skip_weight,
        edge_renderer=edge_renderer,
        label_mode=label_mode,
    )
    Image.fromarray(rgba).save(output_path)
    return output_path


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
) -> _NetworkState:
    if window_steps < 1:
        raise ValueError("window_steps must be >= 1")
    steps = np.asarray(trace["steps"])
    matches = np.flatnonzero(steps == step)
    if matches.size == 0:
        raise ValueError(f"step {step} not found in trace")
    row_index = int(matches[0])
    start = max(0, row_index - window_steps + 1)
    stop = row_index + 1
    return _NetworkState(
        inputs=np.mean(trace["observations"][start:stop], axis=0, dtype=np.float32),
        h1=np.mean(trace["h1"][start:stop], axis=0, dtype=np.float32),
        h2=np.mean(trace["h2"][start:stop], axis=0, dtype=np.float32),
        q_values=np.mean(trace["q_values"][start:stop], axis=0, dtype=np.float32),
        action=int(trace["actions"][row_index]),
    )


def _diff_state(from_state: _NetworkState, to_state: _NetworkState) -> _NetworkState:
    return _NetworkState(
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
    weights = np.asarray([abs(edge.weight) for edge in layout.edges], dtype=np.float32)
    return {
        "input": np.percentile(input_abs, 95, axis=0).astype(float),
        "hidden": float(np.percentile(hidden_values, 95)),
        "output": float(np.percentile(output_abs, 95)),
        "activation": float(np.percentile(np.concatenate([input_abs.ravel(), hidden_values]), 95)),
        "weight": float(np.percentile(weights, 95)) if weights.size else 1.0,
    }


def _render_diff_layout_rgba(
    layout: NetworkLayout, diff_state: _NetworkState, *, scales: Mapping[str, Any], width: int, height: int
) -> np.ndarray:
    weight_scale = _scale_value(
        scales, "weight", max((abs(edge.weight) for edge in layout.edges), default=0.0)
    )
    activation_scale = _scale_value(scales, "activation", _max_source_magnitude(layout.edges, diff_state))
    edge_scale = activation_scale * weight_scale
    fallback_scales = _node_fallback_scales(diff_state)

    def node_fill(node: Node) -> tuple[int, int, int, int]:
        value = _node_value(node, diff_state)
        if node.layer == "in":
            scale = _input_scale(scales, node.index, fallback_scales["input"])
        elif node.layer in {"h1", "h2"}:
            scale = _scale_value(scales, "hidden", fallback_scales["hidden"])
        elif node.layer == "out":
            scale = _scale_value(scales, "output", fallback_scales["output"])
        else:
            scale = 0.0
        return (*color_scheme.signed_color(value, scale), color_scheme.alpha(value, scale))

    def edge_style(edge: Edge) -> _EdgeStyle | None:
        edge_delta = _source_value(edge, diff_state) * edge.weight
        return _EdgeStyle(
            fill=(
                *color_scheme.signed_color(edge_delta, edge_scale),
                color_scheme.alpha(edge_delta, edge_scale),
            ),
            nominal_width=color_scheme.edge_width(edge.weight, weight_scale),
        )

    return _render_layout_rgba(
        layout,
        width=width,
        height=height,
        node_fill=node_fill,
        node_outline=_default_node_outline,
        edge_style=edge_style,
        edge_renderer="pillow",
        label_mode="indices",
    )
