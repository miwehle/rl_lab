import numpy as np
import pytest
import torch
import gymnasium as gym

from nn_viz.layout import Edge, NetworkLayout, Node
from nn_viz.video import (
    LiveOverlayAverager,
    LiveOverlayState,
    _crop_to_visible_alpha,
    _edge_signal_values,
    _node_scale,
    compose_bottom_overlay,
    draw_step_label,
    record_network_overlay_video,
    render_live_layout_rgba,
)


class FakeEnvFactory:
    def __init__(self, env):
        self.env = env
        self.calls = []

    def make_env(self, world, render_mode=None):
        self.calls.append((world, render_mode))
        return self.env


class FakeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.actions = []
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        return np.arange(10, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action)
        self.step_count += 1
        observation = np.arange(10, dtype=np.float32) + self.step_count
        terminated = self.step_count >= 2
        return observation, 0.0, terminated, False, {}

    def render(self):
        return np.zeros((24, 32, 3), dtype=np.uint8)

    def close(self):
        pass


class FakeRecordVideo:
    def __init__(self, env, *, video_folder, name_prefix, **_kwargs):
        self.env = env
        self.path = video_folder / f"{name_prefix}-episode-0.mp4" if hasattr(video_folder, "__truediv__") else None
        self.video_folder = video_folder
        self.name_prefix = name_prefix

    def reset(self, *, seed=None):
        observation = self.env.reset(seed=seed)
        self.env.render()
        return observation

    def step(self, action):
        return self.env.step(action)

    def _capture_frame(self):
        self.env.render()

    def close(self):
        from pathlib import Path

        path = Path(self.video_folder) / f"{self.name_prefix}-episode-0.mp4"
        path.write_bytes(b"video")
        self.env.close()


class TinyQNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 2)
        self.layer2 = torch.nn.Linear(2, 3)
        self.layer3 = torch.nn.Linear(3, 4)
        with torch.no_grad():
            self.layer1.weight.zero_()
            self.layer1.bias.copy_(torch.tensor([1.0, 2.0]))
            self.layer2.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
            self.layer2.bias.zero_()
            self.layer3.weight.zero_()
            self.layer3.bias.copy_(torch.tensor([0.1, 0.4, 0.2, 0.3]))


class MinimalLayout:
    pass


def minimal_live_layout():
    return NetworkLayout(
        nodes=(
            Node("out", 1, "left", 0.0, 0.0, 0.0),
            Node("out", 2, "up", 1.0, 0.0, 0.0),
            Node("h2", 0, "H2-0", 0.0, 1.0, 0.0, 1),
            Node("h1", 0, "H1-0", 0.0, 2.0, 0.0),
            Node("in", 0, "x", 0.0, 3.0, 0.0),
        ),
        edges=(
            Edge("in", 0, "h1", 0, 2.0, 0.0, 0.0),
            Edge("h1", 0, "h2", 0, -3.0, 0.0, 0.0),
            Edge("h2", 0, "out", 1, 4.0, 0.0, 0.0),
        ),
    )


def test_compose_bottom_overlay_blends_only_bottom_band():
    frame = np.full((4, 3, 3), 100, dtype=np.uint8)
    overlay = np.zeros((2, 3, 4), dtype=np.uint8)
    overlay[:, :, 0] = 200
    overlay[:, :, 3] = 255

    composed = compose_bottom_overlay(frame, overlay, alpha=0.5)

    assert composed.dtype == np.uint8
    np.testing.assert_array_equal(composed[:2], frame[:2])
    assert np.all(composed[2:, :, 0] == 150)
    assert np.all(composed[2:, :, 1:] == 50)


def test_compose_bottom_overlay_requires_matching_width():
    frame = np.zeros((4, 3, 3), dtype=np.uint8)
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="overlay width"):
        compose_bottom_overlay(frame, overlay, alpha=0.5)


def test_crop_to_visible_alpha_removes_transparent_margins():
    rgba = np.zeros((5, 6, 4), dtype=np.uint8)
    rgba[1:4, 2:5, 3] = 255

    cropped = _crop_to_visible_alpha(rgba)

    assert cropped.shape == (3, 3, 4)
    assert np.all(cropped[:, :, 3] == 255)


def test_draw_step_label_changes_frame_without_changing_shape():
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    labeled = draw_step_label(frame, 7)

    assert labeled.shape == frame.shape
    assert labeled.dtype == np.uint8
    assert np.any(labeled != frame)


def test_live_overlay_averager_uses_growing_then_rolling_window():
    averager = LiveOverlayAverager(window_steps=2)

    first = averager.update(
        np.array([2.0, -4.0]),
        np.array([1.0]),
        np.array([2.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        3,
    )
    second = averager.update(
        np.array([10.0, 0.0]),
        np.array([5.0]),
        np.array([6.0]),
        np.array([4.0, 5.0, 6.0, 7.0]),
        2,
    )
    third = averager.update(
        np.array([20.0, -8.0]),
        np.array([9.0]),
        np.array([10.0]),
        np.array([8.0, 9.0, 10.0, 11.0]),
        1,
    )

    np.testing.assert_allclose(first.input_abs, [2.0, 4.0])
    np.testing.assert_allclose(second.input_abs, [6.0, 2.0])
    np.testing.assert_allclose(second.h1, [3.0])
    np.testing.assert_allclose(second.h2, [4.0])
    np.testing.assert_allclose(second.q_values, [2.0, 3.0, 4.0, 5.0])
    assert second.action == 2
    np.testing.assert_allclose(third.input_abs, [15.0, 4.0])
    np.testing.assert_allclose(third.h1, [7.0])
    np.testing.assert_allclose(third.h2, [8.0])
    np.testing.assert_allclose(third.q_values, [6.0, 7.0, 8.0, 9.0])
    assert third.action == 1


def test_edge_signal_values_use_source_signal_times_abs_weight():
    layout = minimal_live_layout()
    state = LiveOverlayState(
        input_abs=np.array([2.0]),
        h1=np.array([3.0]),
        h2=np.array([4.0]),
        q_values=np.array([0.0, 1.0, 2.0, 3.0]),
        action=1,
    )

    values = _edge_signal_values(layout.edges, state)

    assert values[layout.edges[0]] == 4.0
    assert values[layout.edges[1]] == 9.0
    assert values[layout.edges[2]] == 16.0


def test_node_scale_prefers_positive_fixed_scale():
    assert _node_scale("h2", {"h2": 7.0}, 3.0) == 7.0
    assert _node_scale("h2", {"h2": 0.0}, 3.0) == 3.0
    assert _node_scale("h2", {"h1": 7.0}, 3.0) == 3.0
    assert _node_scale("h2", None, 3.0) == 3.0


def test_render_live_layout_rgba_returns_nonblank_overlay():
    state = LiveOverlayState(
        input_abs=np.array([2.0]),
        h1=np.array([3.0]),
        h2=np.array([4.0]),
        q_values=np.array([0.0, 1.0, 2.0, 3.0]),
        action=1,
    )

    rgba = render_live_layout_rgba(minimal_live_layout(), state, width=240, height=120)

    assert rgba.shape == (120, 240, 4)
    assert rgba.dtype == np.uint8
    assert np.any(rgba[:, :, 3] > 0)


def test_record_network_overlay_video_writes_trace_and_summary(monkeypatch, tmp_path):
    import nn_viz.video as video

    live_states = []
    live_node_scales = []
    static_render_count = 0

    def static_overlay(_layout, *, width, height):
        nonlocal static_render_count
        static_render_count += 1
        return np.zeros((height, width, 4), dtype=np.uint8)

    monkeypatch.setattr(video, "RecordVideo", FakeRecordVideo)
    monkeypatch.setattr(video, "render_layout_rgba", static_overlay)
    monkeypatch.setattr(
        video,
        "render_live_layout_rgba",
        lambda _layout, state, *, width, height, node_scales=None: live_states.append(state)
        or live_node_scales.append(node_scales)
        or np.zeros((height, width, 4), dtype=np.uint8),
    )
    env = FakeEnv()
    output_path = tmp_path / "earth_seed_0_nn_overlay.mp4"

    recorded_path = record_network_overlay_video(
        TinyQNet(),
        FakeEnvFactory(env),
        MinimalLayout(),
        world="earth",
        seed=123,
        output_path=output_path,
        max_steps=3,
        live_overlay=True,
        live_window_steps=2,
        live_node_scales={"h1": 10.0, "h2": 20.0},
    )

    assert recorded_path == output_path
    assert output_path.read_bytes() == b"video"
    trace = np.load(tmp_path / "earth_seed_0_nn_overlay_trace.npz")
    assert trace["steps"].tolist() == [0, 1]
    assert trace["observations"].shape == (2, 10)
    assert trace["h1"].shape == (2, 2)
    assert trace["h2"].shape == (2, 3)
    assert trace["q_values"].shape == (2, 4)
    np.testing.assert_array_equal(trace["actions"], np.argmax(trace["q_values"], axis=1))

    summary_rows = (tmp_path / "earth_seed_0_nn_overlay_trace_summary.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert summary_rows[0] == "step,action,q_left,q_up,q_noop,q_right"
    assert len(summary_rows) == 3
    assert summary_rows[1].startswith("0,left,")
    assert live_states
    assert static_render_count == 0
    assert live_states[0].action == -1
    np.testing.assert_allclose(live_states[-1].input_abs[:3], [0.5, 1.5, 2.5])
    assert live_node_scales[0] == {"h1": 10.0, "h2": 20.0}
