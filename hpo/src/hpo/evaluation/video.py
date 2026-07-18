"""Record and inspect videos for saved HPO checkpoints."""

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from gymnasium.envs.box2d import lunar_lander
from gymnasium.utils import seeding
from gymnasium.wrappers import RecordVideo

from dqn.model import DQN
from dqn.training import ModelFactory, resolve_device
from hpo.checkpointing import checkpoint_metadata as load_checkpoint_metadata
from hpo.checkpointing import load_checkpoint
from hpo.evaluation.rendering.solar_system_lander import RenderConfig, wrap_env


_FINAL_HOLD_FRAMES = 30
_GOOGLE_DRIVE_MOUNT_CHECKED = False


@dataclass(frozen=True)
class InfraCfg:
    """Infrastructure conventions for recording videos from archived checkpoints."""

    drive_study_dir: Path = Path("/content/drive/MyDrive/rl_lab/hpo")
    best_checkpoints_dir: str = "best_checkpoints"
    videos_dir: str = "videos"
    checkpoint_name: str = "best_eval_checkpoint.pt"
    checkpoint_metadata_name: str = "best_eval_checkpoint.json"

    def checkpoint_dir(self, study_name: str) -> Path:
        return self.drive_study_dir / self.best_checkpoints_dir / study_name

    def checkpoint_path(self, study_name: str) -> Path:
        return self.checkpoint_dir(study_name) / self.checkpoint_name

    def checkpoint_metadata_path(self, study_name: str) -> Path:
        return self.checkpoint_dir(study_name) / self.checkpoint_metadata_name

    def video_dir(self, study_name: str) -> Path:
        return self.drive_study_dir / self.videos_dir / study_name

    def prepare(self) -> None:
        _mount_google_drive_if_available()
        self.drive_study_dir.mkdir(parents=True, exist_ok=True)


def record_video(
    model_factory: ModelFactory,
    env,
    *,
    study_name: str,
    seed: int | None = None,
    max_steps: int = 1_000,
    render_cfg: RenderConfig | None = None,
    device=None,
    cfg: InfraCfg = InfraCfg(),
) -> Path:
    """Record one greedy episode from the conventional archived checkpoint."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    cfg.prepare()
    checkpoint_path = cfg.checkpoint_path(study_name)
    output_dir = cfg.video_dir(study_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _record_checkpoint_video(
        checkpoint_path,
        model_factory,
        env,
        seed=seed,
        max_steps=max_steps,
        render_cfg=render_cfg,
        device=device,
        output_dir=output_dir,
    )


def checkpoint_metadata(study_name: str, *, cfg: InfraCfg = InfraCfg()) -> dict[str, Any]:
    """Return metadata for the conventional archived checkpoint."""
    cfg.prepare()
    return json.loads(cfg.checkpoint_metadata_path(study_name).read_text(encoding="utf-8"))


def video_conditions_table(environment_factory, worlds: Iterable[str], seeds: Iterable[int]):
    """Return weather and initial-force conditions in record-video order."""
    rows = []
    for world in worlds:
        for seed in seeds:
            env = environment_factory.make_env(world)
            try:
                env.reset(seed=seed)
                wind, turbulence = env._weather
                fx, fy = _initial_force(seed)
                rows.append(
                    {
                        "world": env.world.name,
                        "seed": seed,
                        "gravity": env.world.gravity,
                        "wind": wind,
                        "turbulence": turbulence,
                        "initial_force_x": fx,
                        "initial_force_y": fy,
                        "initial_force_abs": math.hypot(fx, fy),
                    }
                )
            finally:
                env.close()

    table = pd.DataFrame(rows).round(2)
    table.insert(0, "nr", range(len(table)))
    return table


def show_video_conditions(environment_factory, worlds: Iterable[str], seeds: Iterable[int]) -> None:
    """Display the video conditions table."""
    from IPython.display import display

    table = video_conditions_table(environment_factory, worlds, seeds)
    display(_video_conditions_style(table))


def _mount_google_drive_if_available() -> None:
    global _GOOGLE_DRIVE_MOUNT_CHECKED
    if _GOOGLE_DRIVE_MOUNT_CHECKED:
        return

    try:
        from google.colab import drive
    except ModuleNotFoundError:
        _GOOGLE_DRIVE_MOUNT_CHECKED = True
        return

    drive.mount("/content/drive")
    _GOOGLE_DRIVE_MOUNT_CHECKED = True


def _record_checkpoint_video(
    checkpoint_path: str | Path,
    model_factory: ModelFactory,
    env,
    *,
    seed: int | None,
    max_steps: int,
    render_cfg: RenderConfig | None,
    device,
    output_dir: str | Path,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    device = resolve_device(device)
    if render_cfg is not None:
        env = wrap_env(env, render_cfg)

    q_net = _q_net_from_env_checkpoint(checkpoint_path, env, device=device, model_factory=model_factory)
    q_net.eval()

    name = _record_video_name(checkpoint_path, env, seed)
    video_env = RecordVideo(
        env,
        video_folder=str(output_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=name,
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

    return _final_video_path(output_dir, name)


def _initial_force(seed: int) -> tuple[float, float]:
    """Reconstruct LunarLander's reset-time random initial force for a seed."""
    rng, _ = seeding.np_random(seed)
    viewport_height = lunar_lander.VIEWPORT_H / lunar_lander.SCALE
    chunks = 11

    # LunarLander.reset draws terrain heights before drawing the initial force.
    rng.uniform(0, viewport_height / 2, size=(chunks + 1,))

    fx = rng.uniform(-lunar_lander.INITIAL_RANDOM, lunar_lander.INITIAL_RANDOM)
    fy = rng.uniform(-lunar_lander.INITIAL_RANDOM, lunar_lander.INITIAL_RANDOM)
    return float(fx), float(fy)


def display_video(video_paths: Iterable[str | Path], nr: int, *, width: int = 720) -> None:
    """Display one recorded video by the table number."""
    from IPython.display import Markdown, Video, display

    path = Path(tuple(video_paths)[nr])
    display(Markdown(f"### {path.name}"))
    display(Video(str(path), embed=True, width=width))


def _video_conditions_style(table: pd.DataFrame):
    float_columns = table.select_dtypes(include="floating").columns
    return table.style.hide(axis="index").format("{:.2f}", subset=float_columns)


def _greedy_action(q_net, observation, device) -> int:
    state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(q_net(state).argmax(dim=1).item())


def _q_net_from_env_checkpoint(path: str | Path, env, *, device, model_factory: ModelFactory):
    n_observations = math.prod(tuple(env.observation_space.shape))
    n_actions = env.action_space.n
    hidden_size = _checkpoint_hidden_size(path)
    q_net = (
        DQN(n_observations, n_actions, hidden_size)
        if model_factory is DQN
        else model_factory(n_observations, n_actions)
    ).to(device)
    load_checkpoint(q_net, path, device)
    return q_net


def _hold_final_frame(env) -> None:
    for _ in range(_FINAL_HOLD_FRAMES):
        env._capture_frame()


def _record_video_name(checkpoint_path: Path, env, seed: int | None) -> str:
    parts = [checkpoint_path.stem]
    world_name = _env_world_name(env)
    if world_name is not None:
        parts.append(world_name)
    if seed is not None:
        parts.append(f"seed_{seed}")
    return "_".join(parts)


def _env_world_name(env) -> str | None:
    current = env
    while current is not None:
        world = getattr(current, "world", None)
        if world is not None:
            name = getattr(world, "name", None)
            if name is not None:
                return str(name)
        current = getattr(current, "env", None)
    return None


def _checkpoint_hidden_size(path: str | Path) -> int:
    metadata = load_checkpoint_metadata(path)
    training_config = metadata.get("training_config", {})
    return int(training_config.get("hidden_size", 128))


def _final_video_path(output_dir: Path, name: str) -> Path:
    raw_path = output_dir / f"{name}-episode-0.mp4"
    final_path = output_dir / f"{name}.mp4"
    if raw_path.exists():
        raw_path.replace(final_path)
    return final_path
