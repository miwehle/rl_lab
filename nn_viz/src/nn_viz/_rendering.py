"""Shared NN layout rendering mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Mapping

import numpy as np

import nn_viz.color_scheme as color_scheme
from nn_viz.layout import Edge, NetworkLayout, Node

_LAYOUT_X_PAD = 0.16
_LAYOUT_TOP_MARGIN_RATIO = 0.18
_LAYOUT_BOTTOM_MARGIN_RATIO = 0.24
_EDGE_SKIP_ACTIVATION_DEFAULT = 0.50
_EDGE_SKIP_WEIGHT_DEFAULT = 0.50
_EDGE_RENDERER_DEFAULT = "pillow"
_LAYER_Y = {"out": -0.125, "h2": 0.125, "h1": 0.375, "in": 0.625}
_INPUT_ORDER = (0, 2, 8, 1, 3, 9, 4, 5, 6, 7)


@dataclass(frozen=True)
class _NetworkState:
    """Averaged NN state used for rendering."""

    inputs: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    q_values: np.ndarray
    action: int


@dataclass(frozen=True)
class _EdgeStyle:
    fill: tuple[int, int, int, int]
    nominal_width: float


def _render_state_layout_rgba(
    layout: NetworkLayout,
    state: _NetworkState,
    *,
    width: int,
    height: int,
    scales: Mapping[str, Any] | None = None,
    edge_skip_activation: float = _EDGE_SKIP_ACTIVATION_DEFAULT,
    edge_skip_weight: float = _EDGE_SKIP_WEIGHT_DEFAULT,
    edge_renderer: str = _EDGE_RENDERER_DEFAULT,
    label_mode: str = "video",
) -> np.ndarray:
    """Render the existing layout as an RGBA overlay for one NN state."""
    weight_scale = _scale_value(scales, "weight", max((abs(edge.weight) for edge in layout.edges), default=0.0))
    activation_scale = _scale_value(scales, "activation", _max_source_magnitude(layout.edges, state))
    fallback_scales = _node_fallback_scales(state)

    def node_fill(node: Node) -> tuple[int, int, int, int]:
        return _node_color(node, state, scales, fallback_scales)

    def node_outline(node: Node, radius: float) -> tuple[tuple[int, int, int, int], int]:
        if node.layer == "out" and node.index == state.action:
            return (250, 204, 21, 255), max(1, int(round(radius / 3)))
        return (17, 24, 39, 255), 1

    def edge_style(edge: Edge) -> _EdgeStyle | None:
        source_value = _source_value(edge, state)
        if _skip_edge(
            source_value, activation_scale, edge.weight, weight_scale, edge_skip_activation, edge_skip_weight
        ):
            return None
        return _EdgeStyle(
            fill=(
                *color_scheme.signed_color(edge.weight, weight_scale),
                color_scheme.alpha(source_value, activation_scale),
            ),
            nominal_width=color_scheme.edge_width(edge.weight, weight_scale),
        )

    return _render_layout_rgba(
        layout,
        width=width,
        height=height,
        node_fill=node_fill,
        node_outline=node_outline,
        edge_style=edge_style,
        edge_renderer=edge_renderer,
        label_mode=label_mode,
    )


def _layout_transform(
    nodes: tuple[Node, ...], *, width: int, height: int
) -> Callable[[float, float], tuple[float, float]]:
    if not nodes:
        return lambda _x, _y: (width / 2, height / 2)
    xs = np.asarray([node.x for node in nodes], dtype=np.float64)
    ys = np.asarray([node.y for node in nodes], dtype=np.float64)
    x_min = float(np.min(xs) - _LAYOUT_X_PAD)
    x_max = float(np.max(xs) + _LAYOUT_X_PAD)
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    margin = max(4.0, min(width, height) * 0.02)
    usable_width = max(1.0, width - margin * 2)
    top_margin = max(margin, height * _LAYOUT_TOP_MARGIN_RATIO)
    bottom_margin = max(margin, height * _LAYOUT_BOTTOM_MARGIN_RATIO)
    usable_height = max(1.0, height - top_margin - bottom_margin)
    y_scale = usable_height / max(1e-9, y_max - y_min)

    def transform(x: float, y: float) -> tuple[float, float]:
        px = margin + (x - x_min) / max(1e-9, x_max - x_min) * usable_width
        py = top_margin + (y - y_min) * y_scale
        return float(px), float(py)

    return transform


def _render_layout_rgba(
    layout: NetworkLayout,
    *,
    width: int,
    height: int,
    node_fill: Callable[[Node], tuple[int, int, int, int]],
    node_outline: Callable[[Node, float], tuple[tuple[int, int, int, int], int]],
    edge_style: Callable[[Edge], _EdgeStyle | None],
    edge_renderer: str,
    label_mode: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    nodes = _display_nodes(layout.nodes)
    node_by_key = {(node.layer, node.index): node for node in nodes}
    transform = _layout_transform(nodes, width=width, height=height)

    image = _draw_styled_edges(
        image, layout.edges, node_by_key, transform, height, edge_style, edge_renderer=edge_renderer
    )
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_styled_nodes(draw, nodes, transform, height, node_fill, node_outline)
    _draw_labels(draw, nodes, transform, height, label_mode=label_mode)
    return np.asarray(image, dtype=np.uint8)


def _display_nodes(nodes: tuple[Node, ...]) -> tuple[Node, ...]:
    output_nodes = [node for node in nodes if node.layer == "out"]
    output_span = _span(output_nodes)
    h2_nodes = [node for node in nodes if node.layer == "h2"]
    h1_nodes = [node for node in nodes if node.layer == "h1"]
    input_nodes = [node for node in nodes if node.layer == "in"]
    hidden_frame = _hidden_frame(h2_nodes, fallback_span=output_span)
    output_ordered = sorted(output_nodes, key=lambda node: (node.x, node.index))
    h2_ordered = tuple(node for group in _output_groups(h2_nodes, output_ordered) for node in group)
    h2_display_nodes = _equidistant_nodes(
        h2_ordered, center=hidden_frame[2], spacing=hidden_frame[3], sort=False
    )
    return (
        _group_start_nodes(output_ordered, h2_nodes, h2_display_nodes, fallback_span=output_span)
        + h2_display_nodes
        + _equidistant_nodes(h1_nodes, center=hidden_frame[2], spacing=hidden_frame[3])
        + _input_nodes(input_nodes, left=hidden_frame[0], right=hidden_frame[1])
    )


def _hidden_frame(
    nodes: list[Node], *, fallback_span: tuple[float, float]
) -> tuple[float, float, float, float]:
    if len(nodes) < 2:
        left, right = fallback_span
        center = (left + right) / 2
        return left, right, center, right - left
    center = _center(nodes)
    original_left, original_right = _span(nodes)
    spacing = (original_right - original_left) / (len(nodes) - 1) / 2
    width = spacing * (len(nodes) - 1)
    return center - width / 2, center + width / 2, center, spacing


def _spread_nodes(nodes: list[Node], *, left: float, right: float) -> tuple[Node, ...]:
    if not nodes:
        return ()
    ordered = sorted(nodes, key=lambda node: (node.x, node.index))
    if len(ordered) == 1:
        xs = [(left + right) / 2]
    else:
        xs = np.linspace(left, right, num=len(ordered))
    return tuple(_display_node(node, x=float(x)) for node, x in zip(ordered, xs))


def _group_start_nodes(
    nodes: list[Node],
    original_source_nodes: list[Node],
    display_source_nodes: tuple[Node, ...],
    *,
    fallback_span: tuple[float, float],
) -> tuple[Node, ...]:
    if not nodes:
        return ()
    if len(display_source_nodes) < len(nodes):
        left, right = _span(list(display_source_nodes)) if display_source_nodes else fallback_span
        return _spread_nodes(nodes, left=left, right=right)
    ordered = sorted(nodes, key=lambda node: (node.x, node.index))
    display_by_index = {node.index: node for node in display_source_nodes}
    blocks = _output_groups(original_source_nodes, ordered)
    xs = [display_by_index[block[0].index].x for block in blocks]
    return tuple(_display_node(node, x=float(x)) for node, x in zip(ordered, xs))


def _output_groups(nodes: list[Node], output_nodes: list[Node]) -> list[list[Node]]:
    groups = [[node for node in nodes if node.output_group == output.index] for output in output_nodes]
    if all(groups):
        return groups
    return [list(group) for group in np.array_split(nodes, len(output_nodes))]


def _equidistant_nodes(
    nodes: list[Node] | tuple[Node, ...], *, center: float, spacing: float, sort: bool = True
) -> tuple[Node, ...]:
    if not nodes:
        return ()
    ordered = sorted(nodes, key=lambda node: (node.x, node.index)) if sort else list(nodes)
    if len(ordered) == 1:
        xs = [center]
    else:
        offsets = (np.arange(len(ordered)) - (len(ordered) - 1) / 2) * spacing
        xs = center + offsets
    return tuple(_display_node(node, x=float(x)) for node, x in zip(ordered, xs))


def _display_node(node: Node, *, x: float) -> Node:
    return replace(node, x=x, y=_LAYER_Y[node.layer])


def _input_nodes(nodes: list[Node], *, left: float, right: float) -> tuple[Node, ...]:
    by_index = {node.index: node for node in nodes}
    ordered = [by_index[index] for index in _INPUT_ORDER if index in by_index]
    if len(ordered) == 1:
        xs = [(left + right) / 2]
    else:
        xs = np.linspace(left, right, num=len(ordered))
    return tuple(_display_node(node, x=float(x)) for node, x in zip(ordered, xs))


def _span(nodes: list[Node]) -> tuple[float, float]:
    if not nodes:
        return 0.0, 1.0
    xs = [node.x for node in nodes]
    return min(xs), max(xs)


def _center(nodes: list[Node]) -> float:
    left, right = _span(nodes)
    return (left + right) / 2


def _draw_styled_edges(
    image,
    edges: tuple[Edge, ...],
    nodes: dict[tuple[str, int], Node],
    transform: Callable[[float, float], tuple[float, float]],
    height: int,
    edge_style: Callable[[Edge], _EdgeStyle | None],
    *,
    edge_renderer: str,
):
    if edge_renderer == "pillow":
        from PIL import ImageDraw

        _draw_styled_edges_pillow(ImageDraw.Draw(image, "RGBA"), edges, nodes, transform, height, edge_style)
        return image
    if edge_renderer == "aggdraw":
        return _draw_styled_edges_aggdraw(image, edges, nodes, transform, height, edge_style)
    raise ValueError("edge_renderer must be 'pillow' or 'aggdraw'")


def _draw_styled_edges_pillow(
    draw,
    edges: tuple[Edge, ...],
    nodes: dict[tuple[str, int], Node],
    transform: Callable[[float, float], tuple[float, float]],
    height: int,
    edge_style: Callable[[Edge], _EdgeStyle | None],
) -> None:
    for edge in edges:
        source = nodes.get((edge.source_layer, edge.source_index))
        target = nodes.get((edge.target_layer, edge.target_index))
        if source is None or target is None:
            continue
        style = edge_style(edge)
        if style is None:
            continue
        sx, sy = transform(source.x, source.y)
        tx, ty = transform(target.x, target.y)
        line_width = max(1, int(round(style.nominal_width * height / 150)))
        draw.line((sx, sy, tx, ty), fill=style.fill, width=line_width)


def _draw_styled_edges_aggdraw(
    image,
    edges: tuple[Edge, ...],
    nodes: dict[tuple[str, int], Node],
    transform: Callable[[float, float], tuple[float, float]],
    height: int,
    edge_style: Callable[[Edge], _EdgeStyle | None],
):
    aggdraw = _load_aggdraw()
    draw = aggdraw.Draw(image)
    for edge in edges:
        source = nodes.get((edge.source_layer, edge.source_index))
        target = nodes.get((edge.target_layer, edge.target_index))
        if source is None or target is None:
            continue
        style = edge_style(edge)
        if style is None:
            continue
        sx, sy = transform(source.x, source.y)
        tx, ty = transform(target.x, target.y)
        line_width = max(1.0, style.nominal_width * height / 150)
        draw.line((sx, sy, tx, ty), aggdraw.Pen(style.fill, line_width))
    draw.flush()

    from PIL import Image

    return Image.fromarray(_unpremultiply_rgba(np.asarray(image, dtype=np.uint8).copy()))


def _draw_styled_nodes(
    draw,
    nodes: tuple[Node, ...],
    transform: Callable[[float, float], tuple[float, float]],
    height: int,
    node_fill: Callable[[Node], tuple[int, int, int, int]],
    node_outline: Callable[[Node, float], tuple[tuple[int, int, int, int], int]],
) -> None:
    radius = max(3.0, height / 46)
    for node in nodes:
        x, y = transform(node.x, node.y)
        fill = node_fill(node)
        outline, outline_width = node_outline(node, radius)
        for offset in range(outline_width, 0, -1):
            draw.ellipse(
                (x - radius - offset, y - radius - offset, x + radius + offset, y + radius + offset),
                fill=outline,
            )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _draw_labels(
    draw,
    nodes: tuple[Node, ...],
    transform: Callable[[float, float], tuple[float, float]],
    height: int,
    *,
    label_mode: str,
) -> None:
    if label_mode not in {"video", "indices"}:
        raise ValueError("label_mode must be 'video' or 'indices'")
    font = _load_font(max(8, height // 36)) if label_mode == "indices" else _load_font(max(16, height // 18))
    for node in nodes:
        x, y = transform(node.x, node.y)
        if label_mode == "indices":
            _draw_centered_text(draw, (x, y + height * 0.07), str(node.index), font, fill=(17, 24, 39, 255))
        elif node.layer == "out":
            _draw_centered_text(draw, (x, y - height * 0.085), node.label, font, fill=(17, 24, 39, 255))
        elif node.layer == "in":
            _draw_centered_text(draw, (x, y + height * 0.085), node.label, font, fill=(17, 24, 39, 255))


def _default_node_outline(_node: Node, _radius: float) -> tuple[tuple[int, int, int, int], int]:
    return (17, 24, 39, 255), 1


def _draw_centered_text(
    draw, center: tuple[float, float], text: str, font, *, fill: tuple[int, int, int, int]
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = center[0] - (bbox[2] - bbox[0]) / 2
    y = center[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font, fill=fill)


def _max_source_magnitude(edges: tuple[Edge, ...], state: _NetworkState) -> float:
    return max((abs(_source_value(edge, state)) for edge in edges), default=0.0)


def _source_value(edge: Edge, state: _NetworkState) -> float:
    if edge.source_layer == "in":
        return _component(state.inputs, edge.source_index)
    if edge.source_layer == "h1":
        return _component(state.h1, edge.source_index)
    if edge.source_layer == "h2":
        return _component(state.h2, edge.source_index)
    return 0.0


def _skip_edge(
    source_value: float,
    activation_scale: float,
    weight: float,
    weight_scale: float,
    edge_skip_activation: float,
    edge_skip_weight: float,
) -> bool:
    return (
        abs(source_value) < edge_skip_activation * activation_scale
        and abs(weight) < edge_skip_weight * weight_scale
    )


def _node_value(node: Node, state: _NetworkState) -> float:
    if node.layer == "in":
        return _component(state.inputs, node.index)
    if node.layer == "h1":
        return _component(state.h1, node.index)
    if node.layer == "h2":
        return _component(state.h2, node.index)
    if node.layer == "out":
        return _component(state.q_values, node.index)
    return 0.0


def _component(values: np.ndarray, index: int) -> float:
    if index < 0 or index >= values.shape[0]:
        return 0.0
    return float(values[index])


def _node_fallback_scales(state: _NetworkState) -> dict[str, float]:
    hidden_values = np.concatenate([state.h1, state.h2])
    return {
        "input": float(np.max(np.abs(state.inputs))) if state.inputs.size else 0.0,
        "hidden": float(np.max(hidden_values)) if hidden_values.size else 0.0,
        "output": float(np.max(np.abs(state.q_values))) if state.q_values.size else 0.0,
    }


def _scale_value(scales: Mapping[str, Any] | None, key: str, fallback: float) -> float:
    if scales is None or key not in scales:
        return fallback
    scale = float(scales[key])
    return scale if scale > 0.0 else fallback


def _input_scale(scales: Mapping[str, Any] | None, index: int, fallback: float) -> float:
    if scales is None or "input" not in scales:
        return fallback
    input_scales = np.asarray(scales["input"], dtype=np.float32)
    if index < 0 or index >= input_scales.shape[0]:
        return fallback
    scale = float(input_scales[index])
    return scale if scale > 0.0 else fallback


def _node_color(
    node: Node,
    state: _NetworkState,
    scales: Mapping[str, Any] | None,
    fallback_scales: Mapping[str, float],
) -> tuple[int, int, int, int]:
    value = _node_value(node, state)
    if node.layer == "in":
        scale = _input_scale(scales, node.index, fallback_scales["input"])
        return (*color_scheme.signed_color(value, scale), color_scheme.alpha(value, scale))
    if node.layer in {"h1", "h2"}:
        scale = _scale_value(scales, "hidden", fallback_scales["hidden"])
        return (*color_scheme.heat_color(value, scale), 255)
    if node.layer == "out":
        scale = _scale_value(scales, "output", fallback_scales["output"])
        return (*color_scheme.signed_color(value, scale), color_scheme.alpha(value, scale))
    return (128, 128, 128, 255)


@lru_cache(maxsize=16)
def _load_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    font_names = ("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf",)
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


@lru_cache(maxsize=1)
def _load_aggdraw():
    try:
        import aggdraw
    except ImportError as exc:
        raise RuntimeError("edge_renderer='aggdraw' requires the aggdraw package") from exc
    return aggdraw


def _unpremultiply_rgba(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3].astype(np.float32)
    visible = alpha > 0
    if not np.any(visible):
        return rgba
    rgb = rgba[:, :, :3].astype(np.float32)
    rgb[visible] = np.minimum(255.0, rgb[visible] * 255.0 / alpha[visible][:, None])
    rgba[:, :, :3] = np.rint(rgb).astype(np.uint8)
    return rgba
