"""Record SolarSystemLander videos with a static NN layout overlay."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

from hpo.evaluation.rendering.solar_system_lander import RenderConfig, wrap_env
from nn_viz.activations import ACTION_LABELS, _forward_activations
from nn_viz.layout import NetworkLayout
from nn_viz.plot import plot_network_layout

_FINAL_HOLD_FRAMES = 30
_CSV_Q_COLUMNS = (("q_left", 1), ("q_up", 2), ("q_noop", 0), ("q_right", 3))


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
    render_cfg: RenderConfig | None = None,
    device: Any = "cpu",
) -> Path:
    """Record one greedy landing video with a static NN layout in the bottom band."""
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
    overlay_env = StaticNetworkOverlayWrapper(
        env,
        layout,
        overlay_height_ratio=overlay_height_ratio,
        overlay_alpha=overlay_alpha,
    )
    overlay_env.set_step(0)
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


class StaticNetworkOverlayWrapper(gym.Wrapper):
    """Blend a cached static network layout into the bottom of rgb_array frames."""

    def __init__(
        self,
        env,
        layout: NetworkLayout,
        *,
        overlay_height_ratio: float,
        overlay_alpha: float,
    ) -> None:
        super().__init__(env)
        if not 0.0 < overlay_height_ratio <= 1.0:
            raise ValueError("overlay_height_ratio must be in (0, 1]")
        if not 0.0 <= overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be in [0, 1]")
        self.layout = layout
        self.overlay_height_ratio = overlay_height_ratio
        self.overlay_alpha = overlay_alpha
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
    """Return an RGB frame with a visible step label in the upper-right corner."""
    from PIL import Image, ImageDraw, ImageFont

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape HxWx3")
    image = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(10, image.height // 34)
    font = ImageFont.truetype("arial.ttf", font_size) if _font_exists("arial.ttf") else ImageFont.load_default(font_size)
    text = f"step: {step:03d}"
    bbox = draw.textbbox((0, 0), text, font=font)
    padding_x = max(10, font_size // 2)
    padding_y = max(6, font_size // 4)
    margin = max(4, min(image.width, image.height) // 50)
    label_width = (bbox[2] - bbox[0]) + padding_x * 2
    label_height = (bbox[3] - bbox[1]) + padding_y * 2
    right = image.width - margin
    left = max(margin, right - label_width)
    top = min(max(48, image.height // 15), max(margin, image.height - label_height - margin))
    bottom = min(image.height - margin, top + label_height)
    draw.rounded_rectangle((left, top, right, bottom), radius=5, fill=(0, 0, 0, 180))
    draw.text((left + padding_x, top + padding_y), text, font=font, fill=(255, 255, 255, 245))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _font_exists(font_name: str) -> bool:
    from PIL import ImageFont

    try:
        ImageFont.truetype(font_name, 12)
    except OSError:
        return False
    return True


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
