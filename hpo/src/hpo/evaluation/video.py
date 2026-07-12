"""Record and inspect videos for saved HPO checkpoints."""

import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from gymnasium.envs.box2d import lunar_lander
from gymnasium.utils import seeding
from gymnasium.wrappers import RecordVideo

from dqn.model import DQN
from dqn.training import ModelFactory, resolve_device
from hpo.evaluation.checkpoint_robustness import q_net_from_checkpoint
from hpo.evaluation.lander_rendering import LanderColors, LanderOverlay, LanderRenderWrapper, LanderSkin

_FINAL_HOLD_FRAMES = 30


def record_checkpoint_video(
    *,
    checkpoint_path: str | Path,
    environment_factory,
    world: str,
    seed: int,
    output_dir: str | Path,
    device=None,
    model_factory: ModelFactory = DQN,
    max_steps: int = 1_000,
    render_colors: LanderColors | None = None,
    render_overlay: LanderOverlay | None = None,
    render_skin: LanderSkin | None = None,
) -> Path:
    """Record one greedy episode for a saved checkpoint."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    device = resolve_device(device)
    world_name = str(world)

    def make_env():
        env = environment_factory.make_env(world_name, render_mode="rgb_array")
        if render_colors is not None or render_overlay is not None or render_skin is not None:
            env = LanderRenderWrapper(env, colors=render_colors, overlay=render_overlay, skin=render_skin)
        return env

    q_net = q_net_from_checkpoint(
        checkpoint_path, make_env=make_env, device=device, model_factory=model_factory
    )
    q_net.eval()

    name = _video_name(checkpoint_path, world_name, seed)
    env = RecordVideo(
        make_env(),
        video_folder=str(output_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=name,
        disable_logger=True,
    )
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(max_steps):
            action = _greedy_action(q_net, observation, device)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                _hold_final_frame(env)
                break
    finally:
        env.close()

    return _final_video_path(output_dir, name)


def record_checkpoint_videos(
    *,
    checkpoint_path: str | Path,
    environment_factory,
    worlds: Iterable[str],
    seeds: Iterable[int],
    output_dir: str | Path,
    device=None,
    model_factory: ModelFactory = DQN,
    max_steps: int = 1_000,
    colors_by_world: Iterable[LanderColors | None] | None = None,
    render_overlay: LanderOverlay | None = None,
    render_skin: LanderSkin | None = None,
    progress: bool = True,
) -> list[Path]:
    """Record one greedy episode for each world/seed pair."""
    worlds = tuple(worlds)
    seeds = tuple(seeds)
    if colors_by_world is None:
        colors_by_world = (None,) * len(worlds)
    else:
        colors_by_world = tuple(colors_by_world)
        if len(colors_by_world) != len(worlds):
            raise ValueError("colors_by_world must have the same length as worlds")

    jobs = [
        (world, seed, render_colors)
        for world, render_colors in zip(worlds, colors_by_world)
        for seed in seeds
    ]
    return [
        record_checkpoint_video(
            checkpoint_path=checkpoint_path,
            environment_factory=environment_factory,
            world=world,
            seed=seed,
            output_dir=output_dir,
            device=device,
            model_factory=model_factory,
            max_steps=max_steps,
            render_colors=render_colors,
            render_overlay=render_overlay,
            render_skin=render_skin,
        )
        for world, seed, render_colors in _with_progress(
            jobs, enabled=progress, total=len(jobs), desc="Recording videos"
        )
    ]


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


def _with_progress(items, *, enabled: bool, total: int, desc: str):
    if not enabled:
        return items
    tqdm = _tqdm()
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc)


def _tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm


def _greedy_action(q_net, observation, device) -> int:
    state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(q_net(state).argmax(dim=1).item())


def _hold_final_frame(env) -> None:
    for _ in range(_FINAL_HOLD_FRAMES):
        env._capture_frame()


def _video_name(checkpoint_path: Path, world: str, seed: int) -> str:
    return f"{checkpoint_path.stem}_{world}_seed_{seed}"


def _final_video_path(output_dir: Path, name: str) -> Path:
    raw_path = output_dir / f"{name}-episode-0.mp4"
    final_path = output_dir / f"{name}.mp4"
    if raw_path.exists():
        raw_path.replace(final_path)
    return final_path
