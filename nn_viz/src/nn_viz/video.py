"""Record SolarSystemLander videos with a static NN layout overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from hpo.evaluation.rendering.solar_system_lander import RenderConfig, wrap_env
from nn_viz.layout import NetworkLayout
from nn_viz.plot import plot_network_layout

_FINAL_HOLD_FRAMES = 30


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
    q_net.eval()

    env = env_factory.make_env(world, render_mode="rgb_array")
    if render_cfg is not None:
        env = wrap_env(env, render_cfg)
    env = StaticNetworkOverlayWrapper(
        env,
        layout,
        overlay_height_ratio=overlay_height_ratio,
        overlay_alpha=overlay_alpha,
    )
    video_env = RecordVideo(
        env,
        video_folder=str(output_path.parent),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=output_path.stem,
        disable_logger=True,
    )
    try:
        observation, _ = video_env.reset(seed=seed)
        for _ in range(max_steps):
            action = _greedy_action(q_net, observation, device)
            observation, _, terminated, truncated, _ = video_env.step(action)
            if terminated or truncated:
                _hold_final_frame(video_env)
                break
    finally:
        video_env.close()

    raw_path = output_path.parent / f"{output_path.stem}-episode-0.mp4"
    if raw_path.exists():
        raw_path.replace(output_path)
    return output_path


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

    def render(self):
        frame = self.env.render()
        if frame is None:
            return None
        height, width = frame.shape[:2]
        overlay_height = max(1, int(round(height * self.overlay_height_ratio)))
        overlay = self._overlay_for(width, overlay_height)
        return compose_bottom_overlay(frame, overlay, alpha=self.overlay_alpha)

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


def _crop_to_visible_alpha(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3]
    visible = np.argwhere(alpha > 0)
    if visible.size == 0:
        return rgba
    top, left = visible.min(axis=0)
    bottom, right = visible.max(axis=0) + 1
    return rgba[top:bottom, left:right]


def _greedy_action(q_net, observation, device) -> int:
    state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(q_net(state).argmax(dim=1).item())


def _hold_final_frame(env) -> None:
    for _ in range(_FINAL_HOLD_FRAMES):
        env._capture_frame()
