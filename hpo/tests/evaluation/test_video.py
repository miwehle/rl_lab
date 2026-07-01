from pathlib import Path

import torch

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


def test_record_checkpoint_videos_records_world_seed_product(monkeypatch, tmp_path):
    calls = []

    def record(**kwargs):
        calls.append((kwargs["world"], kwargs["seed"]))
        return tmp_path / f"{kwargs['world']}_{kwargs['seed']}.mp4"

    monkeypatch.setattr(video, "record_checkpoint_video", record)

    paths = video.record_checkpoint_videos(
        checkpoint_path="checkpoint.pt",
        environment_factory=object(),
        worlds=["earth", "venus"],
        seeds=(seed for seed in [1, 2]),
        output_dir=tmp_path,
    )

    assert calls == [("earth", 1), ("earth", 2), ("venus", 1), ("venus", 2)]
    assert paths == [
        tmp_path / "earth_1.mp4",
        tmp_path / "earth_2.mp4",
        tmp_path / "venus_1.mp4",
        tmp_path / "venus_2.mp4",
    ]
