"""Record SolarSystemLander videos with an NN state overlay."""

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
from nn_viz.activations import ACTION_LABELS, forward_activations
from nn_viz._edges import (
    EDGE_CONTRIBUTORS_PER_TARGET_DEFAULT,
    network_edges_from_q_net,
    scales_with_edge_weight_scale,
    select_edges_by_target_contributors,
    weight_arrays_from_q_net,
)
from nn_viz.layout import NetworkLayout
from nn_viz._rendering import (
    EDGE_RENDERER_DEFAULT,
    NetworkState,
    load_font,
    render_state_layout_rgba,
)

_FINAL_HOLD_FRAMES = 30
_CSV_Q_COLUMNS = (("q_left", 1), ("q_up", 2), ("q_noop", 0), ("q_right", 3))
_WINDOW_STEPS_DEFAULT = 100


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
    edge_contributors_per_target: int = EDGE_CONTRIBUTORS_PER_TARGET_DEFAULT,
    edge_renderer: str = EDGE_RENDERER_DEFAULT,
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
    all_edges = network_edges_from_q_net(q_net)
    render_scales = scales_with_edge_weight_scale(scales, all_edges)

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
            lambda width, height: render_state_layout_rgba(
                NetworkLayout(
                    layout.nodes,
                    select_edges_by_target_contributors(
                        all_edges,
                        averager.state if averager.state is not None else initial_state,
                        edge_contributors_per_target,
                    ),
                ),
                averager.state if averager.state is not None else initial_state,
                width=width,
                height=height,
                scales=render_scales,
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
    trace = _VideoTrace(weights=weight_arrays_from_q_net(q_net))
    try:
        observation, _ = video_env.reset(seed=seed)
        for step in range(max_steps):
            h1, h2, q_values = forward_activations(q_net, observation, device)
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

    weights: dict[str, np.ndarray]
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
            **self.weights,
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
        self.state: NetworkState | None = None

    def update(
        self, observation: np.ndarray, h1: np.ndarray, h2: np.ndarray, q_values: np.ndarray, action: int
    ) -> NetworkState:
        self._inputs.append(np.asarray(observation, dtype=np.float32))
        self._h1.append(np.asarray(h1, dtype=np.float32))
        self._h2.append(np.asarray(h2, dtype=np.float32))
        self._q_values.append(np.asarray(q_values, dtype=np.float32))
        self.state = NetworkState(
            inputs=_mean(self._inputs),
            h1=_mean(self._h1),
            h2=_mean(self._h2),
            q_values=_mean(self._q_values),
            action=action,
        )
        return self.state


def _initial_state(q_net) -> NetworkState:
    h1_size = int(q_net.layer1.out_features)
    h2_size = int(q_net.layer2.out_features)
    action_count = int(q_net.layer3.out_features)
    return NetworkState(
        inputs=np.zeros(int(q_net.layer1.in_features), dtype=np.float32),
        h1=np.zeros(h1_size, dtype=np.float32),
        h2=np.zeros(h2_size, dtype=np.float32),
        q_values=np.zeros(action_count, dtype=np.float32),
        action=-1,
    )


def _mean(values: deque[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(values), axis=0, dtype=np.float32)


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
    font = load_font(font_size, bold=True)
    text = f"step: {step:03d}"
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (image.width - (bbox[2] - bbox[0])) / 2
    y = max(8, image.height // 50)
    shadow_offset = max(1, round(font_size / 18))
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _hold_final_frame(env) -> None:
    for _ in range(_FINAL_HOLD_FRAMES):
        env._capture_frame()
