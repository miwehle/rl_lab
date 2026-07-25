"""Record SolarSystemLander videos with an NN state overlay."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

from hpo.evaluation.rendering.solar_system_lander import RenderConfig, wrap_env
import nn_viz.color_scheme as color_scheme
from nn_viz.activations import ACTION_LABELS, _forward_activations
from nn_viz.layout import Edge, NetworkLayout, Node
from nn_viz.plot import _display_nodes

_FINAL_HOLD_FRAMES = 30
_CSV_Q_COLUMNS = (("q_left", 1), ("q_up", 2), ("q_noop", 0), ("q_right", 3))
_WINDOW_STEPS_DEFAULT = 100
_LAYOUT_X_PAD = 0.16
_LAYOUT_TOP_MARGIN_RATIO = 0.18
_LAYOUT_BOTTOM_MARGIN_RATIO = 0.24
_EDGE_SKIP_ACTIVATION_DEFAULT = 0.50
_EDGE_SKIP_WEIGHT_DEFAULT = 0.50
_EDGE_RENDERER_DEFAULT = "pillow"


def record_video(
    q_net,
    env_factory: Any,
    layout: NetworkLayout,
    *,
    world: str,
    seed: int,
    output_path: str | Path,
    max_steps: int = 1000,
    overlay_height_ratio: float = 0.32,
    overlay_alpha: float = 0.70,
    window_steps: int = _WINDOW_STEPS_DEFAULT,
    scales: Mapping[str, Any] | None = None,
    edge_skip_activation: float = _EDGE_SKIP_ACTIVATION_DEFAULT,
    edge_skip_weight: float = _EDGE_SKIP_WEIGHT_DEFAULT,
    edge_renderer: str = _EDGE_RENDERER_DEFAULT,
    render_cfg: RenderConfig | None = None,
    device: Any = "cpu",
) -> Path:
    """Record one greedy landing video with an NN layout in the bottom band.

    Directly called from the video notebook.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_path.with_name(f"{output_path.stem}_trace.npz")
    summary_path = output_path.with_name(f"{output_path.stem}_trace_summary.csv")
    q_net.eval()

    env = env_factory.make_env(world, render_mode="rgb_array")
    if render_cfg is not None:
        env = wrap_env(env, render_cfg)
    averager = _StateAverager(window_steps)
    initial_state = _initial_state(q_net)
    overlay_env = _NetworkOverlayWrapper(
        env,
        overlay_height_ratio=overlay_height_ratio,
        overlay_alpha=overlay_alpha,
        overlay_provider=(
            lambda width, height: _render_state_layout_rgba(
                layout,
                averager.state if averager.state is not None else initial_state,
                width=width,
                height=height,
                scales=scales,
                edge_skip_activation=edge_skip_activation,
                edge_skip_weight=edge_skip_weight,
                edge_renderer=edge_renderer,
            )
        ),
    )
    video_env = RecordVideo(
        overlay_env,
        video_folder=str(output_path.parent),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=output_path.stem,
        disable_logger=True,
    )
    trace = _VideoTrace()
    try:
        observation, _ = video_env.reset(seed=seed)
        for step in range(max_steps):
            h1, h2, q_values = _forward_activations(q_net, observation, device)
            action = int(np.argmax(q_values))
            trace.append(step, observation, action, h1, h2, q_values)
            averager.update(observation, h1, h2, q_values, action)
            overlay_env.set_step(step)
            observation, _, terminated, truncated, _ = video_env.step(action)
            if terminated or truncated:
                _hold_final_frame(video_env)
                break
    finally:
        video_env.close()

    trace.save(trace_path)
    trace.save_summary(summary_path)

    raw_path = output_path.parent / f"{output_path.stem}-episode-0.mp4"
    if raw_path.exists():
        raw_path.replace(output_path)
    return output_path


@dataclass
class _VideoTrace:
    """Per-step NN state collected while recording one video."""

    steps: list[int] = field(default_factory=list)
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    h1: list[np.ndarray] = field(default_factory=list)
    h2: list[np.ndarray] = field(default_factory=list)
    q_values: list[np.ndarray] = field(default_factory=list)

    def append(
        self,
        step: int,
        observation: np.ndarray,
        action: int,
        h1: np.ndarray,
        h2: np.ndarray,
        q_values: np.ndarray,
    ) -> None:
        self.steps.append(step)
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(action)
        self.h1.append(np.asarray(h1, dtype=np.float32))
        self.h2.append(np.asarray(h2, dtype=np.float32))
        self.q_values.append(np.asarray(q_values, dtype=np.float32))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "steps": np.asarray(self.steps, dtype=np.int64),
            "observations": np.vstack(self.observations).astype(np.float32, copy=False),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "h1": np.vstack(self.h1).astype(np.float32, copy=False),
            "h2": np.vstack(self.h2).astype(np.float32, copy=False),
            "q_values": np.vstack(self.q_values).astype(np.float32, copy=False),
        }

    def save(self, path: Path) -> None:
        np.savez(path, **self.arrays())

    def save_summary(self, path: Path) -> None:
        arrays = self.arrays()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "action", *(name for name, _ in _CSV_Q_COLUMNS)])
            for row_index, step in enumerate(arrays["steps"]):
                q_values = arrays["q_values"][row_index]
                writer.writerow(
                    [
                        int(step),
                        ACTION_LABELS[int(arrays["actions"][row_index])],
                        *(f"{float(q_values[action_index]):.6g}" for _, action_index in _CSV_Q_COLUMNS),
                    ]
                )


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


class _StateAverager:
    """Rolling mean for per-step NN values shown in the moving video."""

    def __init__(self, window_steps: int) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        self.window_steps = window_steps
        self._inputs: deque[np.ndarray] = deque(maxlen=window_steps)
        self._h1: deque[np.ndarray] = deque(maxlen=window_steps)
        self._h2: deque[np.ndarray] = deque(maxlen=window_steps)
        self._q_values: deque[np.ndarray] = deque(maxlen=window_steps)
        self.state: _NetworkState | None = None

    def update(
        self, observation: np.ndarray, h1: np.ndarray, h2: np.ndarray, q_values: np.ndarray, action: int
    ) -> _NetworkState:
        self._inputs.append(np.asarray(observation, dtype=np.float32))
        self._h1.append(np.asarray(h1, dtype=np.float32))
        self._h2.append(np.asarray(h2, dtype=np.float32))
        self._q_values.append(np.asarray(q_values, dtype=np.float32))
        self.state = _NetworkState(
            inputs=_mean(self._inputs),
            h1=_mean(self._h1),
            h2=_mean(self._h2),
            q_values=_mean(self._q_values),
            action=action,
        )
        return self.state


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


def _load_trace_state(trace_path: str | Path, *, step: int, window_steps: int = 1) -> _NetworkState:
    """Load one raw or backward-window-mean NN state from a saved trace."""
    with np.load(trace_path) as trace:
        return _trace_state_from_arrays(trace, step=step, window_steps=window_steps)


def render_trace_step_png(
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
    """Render the NN state for one trace step, averaged over a backward step window."""
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


def render_trace_diff_png(
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
    """Render to-window minus from-window NN differences from one saved trace."""
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


def _initial_state(q_net) -> _NetworkState:
    h1_size = int(q_net.layer1.out_features)
    h2_size = int(q_net.layer2.out_features)
    action_count = int(q_net.layer3.out_features)
    return _NetworkState(
        inputs=np.zeros(int(q_net.layer1.in_features), dtype=np.float32),
        h1=np.zeros(h1_size, dtype=np.float32),
        h2=np.zeros(h2_size, dtype=np.float32),
        q_values=np.zeros(action_count, dtype=np.float32),
        action=-1,
    )


def _mean(values: deque[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(values), axis=0, dtype=np.float32)


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


class _NetworkOverlayWrapper(gym.Wrapper):
    """Blend an NN overlay into the bottom of rgb_array frames."""

    def __init__(
        self,
        env,
        *,
        overlay_height_ratio: float,
        overlay_alpha: float,
        overlay_provider: Callable[[int, int], np.ndarray],
    ) -> None:
        super().__init__(env)
        if not 0.0 < overlay_height_ratio <= 1.0:
            raise ValueError("overlay_height_ratio must be in (0, 1]")
        if not 0.0 <= overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be in [0, 1]")
        self.overlay_height_ratio = overlay_height_ratio
        self.overlay_alpha = overlay_alpha
        self.overlay_provider = overlay_provider
        self._step: int | None = None

    def set_step(self, step: int | None) -> None:
        self._step = step

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
        height, width = frame.shape[:2]
        overlay_height = max(1, int(round(height * self.overlay_height_ratio)))
        overlay = self._overlay_for(width, overlay_height)
        composed = _compose_bottom_overlay(frame, overlay, alpha=self.overlay_alpha)
        if self._step is not None:
            return _draw_step_label(composed, self._step)
        return composed

    def _overlay_for(self, width: int, height: int) -> np.ndarray:
        return self.overlay_provider(width, height)


def _compose_bottom_overlay(frame: np.ndarray, overlay_rgba: np.ndarray, *, alpha: float) -> np.ndarray:
    """Return an RGB frame with an RGBA overlay blended into its bottom band."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape HxWx3")
    if overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4:
        raise ValueError("overlay_rgba must have shape HxWx4")
    if overlay_rgba.shape[1] != frame.shape[1]:
        raise ValueError("overlay width must match frame width")
    if overlay_rgba.shape[0] > frame.shape[0]:
        raise ValueError("overlay height must not exceed frame height")

    output = frame.astype(np.float32, copy=True)
    overlay = overlay_rgba.astype(np.float32, copy=False)
    overlay_height = overlay.shape[0]
    overlay_rgb = overlay[:, :, :3]
    overlay_alpha = (overlay[:, :, 3:4] / 255.0) * alpha
    output[-overlay_height:, :, :] = overlay_rgb * overlay_alpha + output[-overlay_height:, :, :] * (
        1.0 - overlay_alpha
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def _draw_step_label(frame: np.ndarray, step: int) -> np.ndarray:
    """Return an RGB frame with a score-style step label in the top center."""
    from PIL import Image, ImageDraw

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape HxWx3")
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    font_size = max(12, image.height // 33)
    font = _load_font(font_size, bold=True)
    text = f"step: {step:03d}"
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (image.width - (bbox[2] - bbox[0])) / 2
    y = max(8, image.height // 50)
    shadow_offset = max(1, round(font_size / 18))
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


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


def _hold_final_frame(env) -> None:
    for _ in range(_FINAL_HOLD_FRAMES):
        env._capture_frame()
