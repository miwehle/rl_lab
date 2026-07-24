"""Record SolarSystemLander videos with a static NN layout overlay."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

from hpo.evaluation.rendering.solar_system_lander import RenderConfig, wrap_env
import nn_viz.color_scheme as color_scheme
from nn_viz.activations import ACTION_LABELS, _forward_activations
from nn_viz.layout import Edge, NetworkLayout, Node
from nn_viz.plot import _display_nodes, plot_network_layout

_FINAL_HOLD_FRAMES = 30
_CSV_Q_COLUMNS = (("q_left", 1), ("q_up", 2), ("q_noop", 0), ("q_right", 3))
_LIVE_WINDOW_STEPS_DEFAULT = 100


def record_network_overlay_video(
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
    live_overlay: bool = False,
    live_window_steps: int = _LIVE_WINDOW_STEPS_DEFAULT,
    live_scales: Mapping[str, Any] | None = None,
    render_cfg: RenderConfig | None = None,
    device: Any = "cpu",
) -> Path:
    """Record one greedy landing video with an NN layout in the bottom band."""
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
    live_averager = LiveOverlayAverager(live_window_steps) if live_overlay else None
    initial_live_state = _initial_live_state(q_net)
    overlay_env = StaticNetworkOverlayWrapper(
        env,
        layout,
        overlay_height_ratio=overlay_height_ratio,
        overlay_alpha=overlay_alpha,
        overlay_provider=(
            lambda width, height: render_live_layout_rgba(
                layout,
                live_averager.state if live_averager is not None and live_averager.state is not None else initial_live_state,
                width=width,
                height=height,
                live_scales=live_scales,
            )
        )
        if live_overlay
        else None,
    )
    video_env = RecordVideo(
        overlay_env,
        video_folder=str(output_path.parent),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=output_path.stem,
        disable_logger=True,
    )
    trace = VideoTrace()
    try:
        observation, _ = video_env.reset(seed=seed)
        for step in range(max_steps):
            h1, h2, q_values = _forward_activations(q_net, observation, device)
            action = int(np.argmax(q_values))
            trace.append(step, observation, action, h1, h2, q_values)
            if live_averager is not None:
                live_averager.update(observation, h1, h2, q_values, action)
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
class VideoTrace:
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
class LiveOverlayState:
    """Averaged NN state used only for live video rendering."""

    inputs: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    q_values: np.ndarray
    action: int


class LiveOverlayAverager:
    """Rolling mean for per-step NN values shown in the moving video."""

    def __init__(self, window_steps: int) -> None:
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        self.window_steps = window_steps
        self._inputs: deque[np.ndarray] = deque(maxlen=window_steps)
        self._h1: deque[np.ndarray] = deque(maxlen=window_steps)
        self._h2: deque[np.ndarray] = deque(maxlen=window_steps)
        self._q_values: deque[np.ndarray] = deque(maxlen=window_steps)
        self.state: LiveOverlayState | None = None

    def update(
        self,
        observation: np.ndarray,
        h1: np.ndarray,
        h2: np.ndarray,
        q_values: np.ndarray,
        action: int,
    ) -> LiveOverlayState:
        self._inputs.append(np.asarray(observation, dtype=np.float32))
        self._h1.append(np.asarray(h1, dtype=np.float32))
        self._h2.append(np.asarray(h2, dtype=np.float32))
        self._q_values.append(np.asarray(q_values, dtype=np.float32))
        self.state = LiveOverlayState(
            inputs=_mean(self._inputs),
            h1=_mean(self._h1),
            h2=_mean(self._h2),
            q_values=_mean(self._q_values),
            action=action,
        )
        return self.state


def render_live_layout_rgba(
    layout: NetworkLayout,
    live_state: LiveOverlayState,
    *,
    width: int,
    height: int,
    live_scales: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Render the existing layout as a dynamic RGBA overlay."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    nodes = _display_nodes(layout.nodes)
    node_by_key = {(node.layer, node.index): node for node in nodes}
    transform = _layout_transform(nodes, width=width, height=height)

    _draw_live_edges(draw, layout.edges, node_by_key, live_state, live_scales, transform, height)
    _draw_live_nodes(draw, nodes, live_state, transform, height, live_scales)
    _draw_live_labels(draw, nodes, live_state, transform, height)
    return np.asarray(image, dtype=np.uint8)


def _initial_live_state(q_net) -> LiveOverlayState:
    h1_size = int(q_net.layer1.out_features)
    h2_size = int(q_net.layer2.out_features)
    action_count = int(q_net.layer3.out_features)
    return LiveOverlayState(
        inputs=np.zeros(int(q_net.layer1.in_features), dtype=np.float32),
        h1=np.zeros(h1_size, dtype=np.float32),
        h2=np.zeros(h2_size, dtype=np.float32),
        q_values=np.zeros(action_count, dtype=np.float32),
        action=-1,
    )


def _mean(values: deque[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(values), axis=0, dtype=np.float32)


def _layout_transform(
    nodes: tuple[Node, ...],
    *,
    width: int,
    height: int,
) -> Callable[[float, float], tuple[float, float]]:
    if not nodes:
        return lambda _x, _y: (width / 2, height / 2)
    xs = np.asarray([node.x for node in nodes], dtype=np.float64)
    ys = np.asarray([node.y for node in nodes], dtype=np.float64)
    x_min = float(np.min(xs) - 0.16)
    x_max = float(np.max(xs) + 0.16)
    y_min = float(np.min(ys) - 0.22)
    y_max = float(np.max(ys) + 0.18)
    margin = max(4.0, min(width, height) * 0.02)
    usable_width = max(1.0, width - margin * 2)
    usable_height = max(1.0, height - margin * 2)

    def transform(x: float, y: float) -> tuple[float, float]:
        px = margin + (x - x_min) / max(1e-9, x_max - x_min) * usable_width
        py = margin + (y - y_min) / max(1e-9, y_max - y_min) * usable_height
        return float(px), float(py)

    return transform


def _draw_live_edges(
    draw,
    edges: tuple[Edge, ...],
    nodes: dict[tuple[str, int], Node],
    live_state: LiveOverlayState,
    live_scales: Mapping[str, Any] | None,
    transform: Callable[[float, float], tuple[float, float]],
    height: int,
) -> None:
    weight_scale = _scale_value(live_scales, "weight", max((abs(edge.weight) for edge in edges), default=0.0))
    activation_scale = _scale_value(live_scales, "activation", _max_source_magnitude(edges, live_state))
    for edge in edges:
        source = nodes.get((edge.source_layer, edge.source_index))
        target = nodes.get((edge.target_layer, edge.target_index))
        if source is None or target is None:
            continue
        sx, sy = transform(source.x, source.y)
        tx, ty = transform(target.x, target.y)
        edge_alpha = color_scheme.alpha(_source_value(edge, live_state), activation_scale)
        nominal_width = color_scheme.edge_width(edge.weight, weight_scale)
        line_width = max(1, int(round(nominal_width * height / 150)))
        draw.line(
            (sx, sy, tx, ty),
            fill=(*color_scheme.signed_color(edge.weight, weight_scale), edge_alpha),
            width=line_width,
        )


def _draw_live_nodes(
    draw,
    nodes: tuple[Node, ...],
    live_state: LiveOverlayState,
    transform: Callable[[float, float], tuple[float, float]],
    height: int,
    live_scales: Mapping[str, Any] | None,
) -> None:
    radius = max(3.0, height / 46)
    fallback_scales = _node_fallback_scales(live_state)
    for node in nodes:
        x, y = transform(node.x, node.y)
        fill = _live_node_color(node, live_state, live_scales, fallback_scales)
        outline = (250, 204, 21, 255) if node.layer == "out" and node.index == live_state.action else (17, 24, 39, 255)
        outline_width = max(1, int(round(radius / 3))) if node.layer == "out" and node.index == live_state.action else 1
        for offset in range(outline_width, 0, -1):
            draw.ellipse((x - radius - offset, y - radius - offset, x + radius + offset, y + radius + offset), fill=outline)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _draw_live_labels(
    draw,
    nodes: tuple[Node, ...],
    live_state: LiveOverlayState,
    transform: Callable[[float, float], tuple[float, float]],
    height: int,
) -> None:
    font = _load_font(max(16, height // 18))
    hidden_font = _load_font(max(6, height // 45))
    for node in nodes:
        x, y = transform(node.x, node.y)
        if node.layer == "out":
            _draw_centered_text(draw, (x, y - height * 0.085), node.label, font, fill=(17, 24, 39, 255))
        elif node.layer in {"h1", "h2"}:
            _draw_centered_text(draw, (x, y + height * 0.055), str(node.index), hidden_font, fill=(17, 24, 39, 255))
        elif node.layer == "in":
            _draw_centered_text(draw, (x, y + height * 0.085), node.label, font, fill=(17, 24, 39, 255))


def _draw_centered_text(draw, center: tuple[float, float], text: str, font, *, fill: tuple[int, int, int, int]) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = center[0] - (bbox[2] - bbox[0]) / 2
    y = center[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font, fill=fill)


def _max_source_magnitude(edges: tuple[Edge, ...], live_state: LiveOverlayState) -> float:
    return max((abs(_source_value(edge, live_state)) for edge in edges), default=0.0)


def _source_value(edge: Edge, live_state: LiveOverlayState) -> float:
    if edge.source_layer == "in":
        return _component(live_state.inputs, edge.source_index)
    if edge.source_layer == "h1":
        return _component(live_state.h1, edge.source_index)
    if edge.source_layer == "h2":
        return _component(live_state.h2, edge.source_index)
    return 0.0


def _node_value(node: Node, live_state: LiveOverlayState) -> float:
    if node.layer == "in":
        return _component(live_state.inputs, node.index)
    if node.layer == "h1":
        return _component(live_state.h1, node.index)
    if node.layer == "h2":
        return _component(live_state.h2, node.index)
    if node.layer == "out":
        return _component(live_state.q_values, node.index)
    return 0.0


def _component(values: np.ndarray, index: int) -> float:
    if index < 0 or index >= values.shape[0]:
        return 0.0
    return float(values[index])


def _node_fallback_scales(live_state: LiveOverlayState) -> dict[str, float]:
    return {
        "input": float(np.max(np.abs(live_state.inputs))) if live_state.inputs.size else 0.0,
        "h1": float(np.max(live_state.h1)) if live_state.h1.size else 0.0,
        "h2": float(np.max(live_state.h2)) if live_state.h2.size else 0.0,
        "output": float(np.max(np.abs(live_state.q_values))) if live_state.q_values.size else 0.0,
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


def _live_node_color(
    node: Node,
    live_state: LiveOverlayState,
    live_scales: Mapping[str, Any] | None,
    fallback_scales: Mapping[str, float],
) -> tuple[int, int, int, int]:
    value = _node_value(node, live_state)
    if node.layer == "in":
        scale = _input_scale(live_scales, node.index, fallback_scales["input"])
        return (*color_scheme.signed_color(value, scale), color_scheme.alpha(value, scale))
    if node.layer in {"h1", "h2"}:
        scale = _scale_value(live_scales, node.layer, fallback_scales[node.layer])
        return (*color_scheme.heat_color(value, scale), 255)
    if node.layer == "out":
        scale = _scale_value(live_scales, "output", fallback_scales["output"])
        return (*color_scheme.signed_color(value, scale), color_scheme.alpha(value, scale))
    return (128, 128, 128, 255)


class StaticNetworkOverlayWrapper(gym.Wrapper):
    """Blend a cached static network layout into the bottom of rgb_array frames."""

    def __init__(
        self,
        env,
        layout: NetworkLayout,
        *,
        overlay_height_ratio: float,
        overlay_alpha: float,
        overlay_provider: Callable[[int, int], np.ndarray] | None = None,
    ) -> None:
        super().__init__(env)
        if not 0.0 < overlay_height_ratio <= 1.0:
            raise ValueError("overlay_height_ratio must be in (0, 1]")
        if not 0.0 <= overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be in [0, 1]")
        self.layout = layout
        self.overlay_height_ratio = overlay_height_ratio
        self.overlay_alpha = overlay_alpha
        self.overlay_provider = overlay_provider
        self._overlay_rgba: np.ndarray | None = None
        self._overlay_size: tuple[int, int] | None = None
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
        composed = compose_bottom_overlay(frame, overlay, alpha=self.overlay_alpha)
        if self._step is not None:
            return draw_step_label(composed, self._step)
        return composed

    def _overlay_for(self, width: int, height: int) -> np.ndarray:
        if self.overlay_provider is not None:
            return self.overlay_provider(width, height)
        size = (width, height)
        if self._overlay_rgba is None or self._overlay_size != size:
            self._overlay_rgba = render_layout_rgba(self.layout, width=width, height=height)
            self._overlay_size = size
        return self._overlay_rgba


def compose_bottom_overlay(frame: np.ndarray, overlay_rgba: np.ndarray, *, alpha: float) -> np.ndarray:
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
    output[-overlay_height:, :, :] = (
        overlay_rgb * overlay_alpha + output[-overlay_height:, :, :] * (1.0 - overlay_alpha)
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def render_layout_rgba(layout: NetworkLayout, *, width: int, height: int) -> np.ndarray:
    """Render the existing static layout plot as an RGBA image with exact pixel size."""
    from PIL import Image
    import matplotlib.pyplot as plt

    fig = plot_network_layout(layout)
    try:
        fig.patch.set_alpha(0.0)
        for ax in fig.axes:
            ax.patch.set_alpha(0.0)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    finally:
        plt.close(fig)

    image = Image.fromarray(_crop_to_visible_alpha(rgba))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    image.thumbnail((width, height), resampling)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    canvas.paste(image, ((width - image.width) // 2, height - image.height), image)
    return np.asarray(canvas, dtype=np.uint8)


def draw_step_label(frame: np.ndarray, step: int) -> np.ndarray:
    """Return an RGB frame with a score-style step label in the top center."""
    from PIL import Image, ImageDraw

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape HxWx3")
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(12, image.height // 33)
    font = _load_font(font_size, bold=True)
    text = f"step: {step:03d}"
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (image.width - (bbox[2] - bbox[0])) / 2
    y = max(8, image.height // 50)
    shadow_offset = max(1, round(font_size / 18))
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 210))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 245))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _load_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    font_names = ("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf",)
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def _crop_to_visible_alpha(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3]
    visible = np.argwhere(alpha > 0)
    if visible.size == 0:
        return rgba
    top, left = visible.min(axis=0)
    bottom, right = visible.max(axis=0) + 1
    return rgba[top:bottom, left:right]


def _hold_final_frame(env) -> None:
    for _ in range(_FINAL_HOLD_FRAMES):
        env._capture_frame()
