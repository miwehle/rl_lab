from pathlib import Path

import pandas as pd
import pytest
import torch
from IPython import display as ipython_display

from hpo.evaluation import video


class FakeEnv:
    world = type("World", (), {"name": "venus"})()

    def __init__(self):
        self.observation_space = type("ObservationSpace", (), {"shape": (2,)})()
        self.action_space = type("ActionSpace", (), {"n": 2})()

    def reset(self, *, seed=None):
        self.seed = seed
        return [0.0, 0.0], {}

    def step(self, action):
        self.action = action
        return [0.0, 0.0], 0.0, True, False, {}

    def close(self):
        pass


class FakeRecordVideo:
    def __init__(self, env, *, video_folder, name_prefix, **_kwargs):
        self.env = env
        self.path = Path(video_folder) / f"{name_prefix}-episode-0.mp4"

    def reset(self, *, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def _capture_frame(self):
        pass

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"video")
        self.env.close()


class FakeQNet(torch.nn.Module):
    def forward(self, state):
        return torch.tensor([[0.0, 1.0]], device=state.device)


def test_record_video_records_one_greedy_episode(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "RecordVideo", FakeRecordVideo)
    monkeypatch.setattr(video, "_q_net_from_env_checkpoint", lambda *_args, **_kwargs: FakeQNet())

    path = video.record_video(
        "trial_0009_eval_best.pt", video.DQN, FakeEnv(), seed=10_000, output_dir=tmp_path
    )

    assert path == tmp_path / "trial_0009_eval_best_venus_seed_10000.mp4"
    assert path.read_bytes() == b"video"


def test_record_video_wraps_env_for_render_config(monkeypatch, tmp_path):
    wrapped_env = FakeEnv()
    calls = []
    recorded_envs = []

    class RecordingRecordVideo(FakeRecordVideo):
        def __init__(self, env, *, video_folder, name_prefix, **kwargs):
            recorded_envs.append(env)
            super().__init__(env, video_folder=video_folder, name_prefix=name_prefix, **kwargs)

    def fake_wrap_env(env, render_cfg):
        calls.append((env, render_cfg))
        return wrapped_env

    monkeypatch.setattr(video, "RecordVideo", RecordingRecordVideo)
    monkeypatch.setattr(video, "wrap_env", fake_wrap_env)
    monkeypatch.setattr(video, "_q_net_from_env_checkpoint", lambda *_args, **_kwargs: FakeQNet())
    env = FakeEnv()
    render_cfg = video.RenderConfig(colors_by_world=(None,))

    video.record_video("trial_0009_eval_best.pt", video.DQN, env, render_cfg=render_cfg, output_dir=tmp_path)

    assert calls == [(env, render_cfg)]
    assert recorded_envs == [wrapped_env]


def test_record_video_rejects_invalid_max_steps(tmp_path):
    with pytest.raises(ValueError, match="max_steps"):
        video.record_video("trial_0009_eval_best.pt", video.DQN, FakeEnv(), max_steps=0, output_dir=tmp_path)


def test_show_video_conditions_formats_floats_with_two_decimals(monkeypatch):
    displayed = []
    table = pd.DataFrame({"nr": [0], "world": ["earth"], "seed": [7], "gravity": [-10.0], "wind": [1.2]})

    monkeypatch.setattr(video, "video_conditions_table", lambda *_args: table)
    monkeypatch.setattr(ipython_display, "display", displayed.append)

    video.show_video_conditions(object(), ["earth"], [7])

    html = displayed[0].to_html()
    assert "-10.00" in html
    assert "1.20" in html
