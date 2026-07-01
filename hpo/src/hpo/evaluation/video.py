"""Record videos for saved HPO checkpoints."""

from collections.abc import Iterable
from pathlib import Path

import torch
from gymnasium.wrappers import RecordVideo

from dqn.model import DQN
from dqn.training import ModelFactory, resolve_device
from hpo.checkpoint_robustness import q_net_from_checkpoint


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
) -> Path:
    """Record one greedy episode for a saved checkpoint."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    device = resolve_device(device)
    world_name = str(world)
    make_env = lambda: environment_factory.make_env(
        world_name,
        render_mode="rgb_array",
    )
    q_net = q_net_from_checkpoint(
        checkpoint_path,
        make_env=make_env,
        device=device,
        model_factory=model_factory,
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
) -> list[Path]:
    """Record one greedy episode for each world/seed pair."""
    worlds = tuple(worlds)
    seeds = tuple(seeds)
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
        )
        for world in worlds
        for seed in seeds
    ]


def _greedy_action(q_net, observation, device) -> int:
    state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(q_net(state).argmax(dim=1).item())


def _video_name(checkpoint_path: Path, world: str, seed: int) -> str:
    return f"{checkpoint_path.stem}_{world}_seed_{seed}"


def _final_video_path(output_dir: Path, name: str) -> Path:
    raw_path = output_dir / f"{name}-episode-0.mp4"
    final_path = output_dir / f"{name}.mp4"
    if raw_path.exists():
        raw_path.replace(final_path)
    return final_path
