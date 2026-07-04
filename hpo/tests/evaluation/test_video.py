from pathlib import Path

import pandas as pd
import pytest
import torch
from IPython import display as ipython_display

from hpo.evaluation import video


class FakeEnv:
    def reset(self, *, seed=None):
        self.seed = seed
        return [0.0, 0.0], {}

    def step(self, action):
        self.action = action
        return [0.0, 0.0], 0.0, True, False, {}

    def close(self):
        pass


class FakeFactory:
    def __init__(self):
        self.calls = []

    def make_env(self, world, *, render_mode=None):
        self.calls.append((world, render_mode))
        return FakeEnv()


class FakeRecordVideo:
    def __init__(self, env, *, video_folder, name_prefix, **_kwargs):
        self.env = env
        self.path = Path(video_folder) / f"{name_prefix}-episode-0.mp4"

    def reset(self, *, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"video")
        self.env.close()


class FakeRenderWrapper:
    def __init__(self, env, *, colors=None, overlay=None):
        self.env = env
        self.colors = colors
        self.overlay = overlay

    def reset(self, *, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class FakeQNet(torch.nn.Module):
    def forward(self, state):
        return torch.tensor([[0.0, 1.0]], device=state.device)


def test_record_checkpoint_video_records_one_greedy_episode(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "RecordVideo", FakeRecordVideo)
    monkeypatch.setattr(video, "q_net_from_checkpoint", lambda *_args, **_kwargs: FakeQNet())

    path = video.record_checkpoint_video(
        checkpoint_path="trial_0009_eval_best.pt",
        environment_factory=FakeFactory(),
        world="venus",
        seed=10_000,
        output_dir=tmp_path,
    )

    assert path == tmp_path / "trial_0009_eval_best_venus_seed_10000.mp4"
    assert path.read_bytes() == b"video"


def test_record_checkpoint_video_wraps_env_when_render_colors_are_given(
    monkeypatch,
    tmp_path,
):
    recorded_envs = []

    class RecordingRecordVideo(FakeRecordVideo):
        def __init__(self, env, *, video_folder, name_prefix, **kwargs):
            recorded_envs.append(env)
            super().__init__(
                env,
                video_folder=video_folder,
                name_prefix=name_prefix,
                **kwargs,
            )

    monkeypatch.setattr(video, "RecordVideo", RecordingRecordVideo)
    monkeypatch.setattr(video, "LanderRenderWrapper", FakeRenderWrapper)
    monkeypatch.setattr(video, "q_net_from_checkpoint", lambda *_args, **_kwargs: FakeQNet())
    colors = video.LanderColors(sky=(1, 2, 3))

    video.record_checkpoint_video(
        checkpoint_path="trial_0009_eval_best.pt",
        environment_factory=FakeFactory(),
        world="venus",
        seed=10_000,
        output_dir=tmp_path,
        render_colors=colors,
    )

    assert isinstance(recorded_envs[0], FakeRenderWrapper)
    assert recorded_envs[0].colors == colors
    assert recorded_envs[0].overlay is None


def test_record_checkpoint_video_wraps_env_when_render_overlay_is_given(
    monkeypatch,
    tmp_path,
):
    recorded_envs = []

    class RecordingRecordVideo(FakeRecordVideo):
        def __init__(self, env, *, video_folder, name_prefix, **kwargs):
            recorded_envs.append(env)
            super().__init__(
                env,
                video_folder=video_folder,
                name_prefix=name_prefix,
                **kwargs,
            )

    monkeypatch.setattr(video, "RecordVideo", RecordingRecordVideo)
    monkeypatch.setattr(video, "LanderRenderWrapper", FakeRenderWrapper)
    monkeypatch.setattr(video, "q_net_from_checkpoint", lambda *_args, **_kwargs: FakeQNet())
    overlay = video.LanderOverlay()

    video.record_checkpoint_video(
        checkpoint_path="trial_0009_eval_best.pt",
        environment_factory=FakeFactory(),
        world="venus",
        seed=10_000,
        output_dir=tmp_path,
        render_overlay=overlay,
    )

    assert isinstance(recorded_envs[0], FakeRenderWrapper)
    assert recorded_envs[0].colors is None
    assert recorded_envs[0].overlay == overlay


def test_record_checkpoint_videos_records_world_seed_product(monkeypatch, tmp_path):
    calls = []

    def record(**kwargs):
        calls.append(
            (
                kwargs["world"],
                kwargs["seed"],
                kwargs["render_colors"],
                kwargs["render_overlay"],
            )
        )
        return tmp_path / f"{kwargs['world']}_{kwargs['seed']}.mp4"

    monkeypatch.setattr(video, "record_checkpoint_video", record)
    earth_colors = video.LanderColors(sky=(1, 2, 3))
    venus_colors = video.LanderColors(sky=(4, 5, 6))
    overlay = video.LanderOverlay()

    paths = video.record_checkpoint_videos(
        checkpoint_path="checkpoint.pt",
        environment_factory=object(),
        worlds=["earth", "venus"],
        seeds=(seed for seed in [1, 2]),
        output_dir=tmp_path,
        colors_by_world=[earth_colors, venus_colors],
        render_overlay=overlay,
    )

    assert calls == [
        ("earth", 1, earth_colors, overlay),
        ("earth", 2, earth_colors, overlay),
        ("venus", 1, venus_colors, overlay),
        ("venus", 2, venus_colors, overlay),
    ]
    assert paths == [
        tmp_path / "earth_1.mp4",
        tmp_path / "earth_2.mp4",
        tmp_path / "venus_1.mp4",
        tmp_path / "venus_2.mp4",
    ]


def test_record_checkpoint_videos_rejects_mismatched_colors_by_world(tmp_path):
    with pytest.raises(ValueError, match="colors_by_world"):
        video.record_checkpoint_videos(
            checkpoint_path="checkpoint.pt",
            environment_factory=object(),
            worlds=["earth", "venus"],
            seeds=[1],
            output_dir=tmp_path,
            colors_by_world=[video.LanderColors()],
        )


def test_show_video_conditions_formats_floats_with_two_decimals(monkeypatch):
    displayed = []
    table = pd.DataFrame(
        {
            "nr": [0],
            "world": ["earth"],
            "seed": [7],
            "gravity": [-10.0],
            "wind": [1.2],
        }
    )

    monkeypatch.setattr(video, "video_conditions_table", lambda *_args: table)
    monkeypatch.setattr(ipython_display, "display", displayed.append)

    video.show_video_conditions(object(), ["earth"], [7])

    html = displayed[0].to_html()
    assert "-10.00" in html
    assert "1.20" in html
