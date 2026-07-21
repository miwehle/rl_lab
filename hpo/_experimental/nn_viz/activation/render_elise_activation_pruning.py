"""Activation-aware pruning diagnostics for Elise-264-GSTP."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "dqn" / "src"))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dqn.model import DQN

from hpo.environments.solar_system_lander.env import DEFAULT_WORLD_MIX, EnvFactory, EnvWrapper, World, WorldConfig
from hpo.evaluation.rendering.solar_system_lander._env_state import _initial_kick

CHECKPOINT_DIR = Path(r"G:\Meine Ablage\rl_lab\hpo\best_checkpoints\solar_system_lander_10d_elise_stp")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_eval_checkpoint.pt"
OUT_DIR = Path(__file__).resolve().parent
EPS = 1e-6


@dataclass(frozen=True)
class Metrics:
    layer: str
    index: int
    active_frequency: float
    mean_activation: float
    max_activation: float
    outgoing_abs_sum: float
    activation_score: float


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_net = _load_model()
    h1, h2, actions, episode_rows = _collect_activations(q_net)
    h1_metrics = _metrics("H1", h1, _outgoing_abs_sum(q_net, "H1"))
    h2_metrics = _metrics("H2", h2, _outgoing_abs_sum(q_net, "H2"))

    _write_metrics(OUT_DIR / "elise_activation_metrics.csv", h1_metrics + h2_metrics)
    _write_summary(OUT_DIR / "elise_activation_pruning_summary.md", h1_metrics, h2_metrics, actions, episode_rows)
    _draw_ranked(OUT_DIR / "elise_activation_ranked_importance.png", h1_metrics, h2_metrics)
    _draw_frequency_histograms(OUT_DIR / "elise_activation_frequency_histograms.png", h1_metrics, h2_metrics)
    print(f"wrote activation-aware pruning diagnostics to: {OUT_DIR}")
    return 0


def _load_model() -> DQN:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    q_net = DQN(n_observations=10, n_actions=4, hidden_size=128)
    q_net.load_state_dict(state)
    q_net.eval()
    return q_net


def _collect_activations(q_net: DQN) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    h1_values: list[np.ndarray] = []
    h2_values: list[np.ndarray] = []
    actions: list[int] = []
    episode_rows: list[dict[str, object]] = []
    factory = EnvFactory("10d", world_mix=DEFAULT_WORLD_MIX)

    for world in [World.MERCURY, World.VENUS, World.EARTH, World.MOON, World.MARS]:
        for seed in range(50):
            _run_episode(factory.make_env(world), q_net, seed, f"default:{world}", h1_values, h2_values, actions, episode_rows)

    for world in [World.EARTH, World.VENUS]:
        _run_episode(factory.make_env(world), q_net, 10014, f"known-hard:{world}", h1_values, h2_values, actions, episode_rows)

    for world in [World.EARTH, World.VENUS]:
        for seed in _strong_downward_kick_seeds(world, count=20):
            _run_episode(_fixed_weather_env(world, wind=20.0, turbulence=2.0), q_net, seed, f"max-weather-downkick:{world}", h1_values, h2_values, actions, episode_rows)

    return np.vstack(h1_values), np.vstack(h2_values), np.array(actions), episode_rows


def _run_episode(env, q_net: DQN, seed: int, scenario: str, h1_values, h2_values, actions, episode_rows) -> None:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    try:
        for steps in range(1, 1001):
            x = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                h1 = F.relu(q_net.layer1(x))
                h2 = F.relu(q_net.layer2(h1))
                action = int(q_net.layer3(h2).argmax(dim=1).item())
            h1_values.append(h1[0].numpy())
            h2_values.append(h2[0].numpy())
            actions.append(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()
    episode_rows.append({"scenario": scenario, "seed": seed, "steps": steps, "score": total_reward})


def _fixed_weather_env(world: World, *, wind: float, turbulence: float):
    config = WorldConfig(world, -9.0 if world == World.VENUS else -10.0, (wind, wind), (turbulence, turbulence))
    return EnvWrapper(gym.make("LunarLander-v3", gravity=config.gravity, enable_wind=True), config, "10d")


def _strong_downward_kick_seeds(world: World, *, count: int) -> list[int]:
    env = _fixed_weather_env(world, wind=20.0, turbulence=2.0)
    env.reset(seed=0)
    mass = env.unwrapped.lander.mass
    env.close()
    ranked = sorted((_initial_kick(seed, mass)[1], seed) for seed in range(5000))
    return [seed for _dy, seed in ranked[:count]]


def _outgoing_abs_sum(q_net: DQN, layer: str) -> np.ndarray:
    if layer == "H1":
        return np.abs(q_net.layer2.weight.detach().numpy()).sum(axis=0)
    return np.abs(q_net.layer3.weight.detach().numpy()).sum(axis=0)


def _metrics(layer: str, activations: np.ndarray, outgoing_abs_sum: np.ndarray) -> list[Metrics]:
    return [
        Metrics(
            layer=layer,
            index=index,
            active_frequency=float(np.mean(values > EPS)),
            mean_activation=float(np.mean(values)),
            max_activation=float(np.max(values)),
            outgoing_abs_sum=float(outgoing_abs_sum[index]),
            activation_score=float(np.mean(values) * outgoing_abs_sum[index]),
        )
        for index, values in enumerate(activations.T)
    ]


def _write_metrics(path: Path, rows: list[Metrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(Metrics.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_summary(path: Path, h1: list[Metrics], h2: list[Metrics], actions: np.ndarray, episodes: list[dict[str, object]]) -> None:
    lines = [
        "# Elise Activation-Aware Pruning Diagnostics",
        "",
        f"Checkpoint: `{CHECKPOINT_PATH}`",
        f"Frames: `{len(actions)}`",
        f"Episodes: `{len(episodes)}`",
        f"Action fractions: noop={np.mean(actions == 0):.3f}, left={np.mean(actions == 1):.3f}, main={np.mean(actions == 2):.3f}, right={np.mean(actions == 3):.3f}",
        "",
        "Metric: `activation_score = mean(ReLU(activation)) * outgoing_abs_sum`.",
        "",
    ]
    lines += _table("H1 strongest active neurons", sorted(h1, key=lambda row: row.activation_score, reverse=True)[:20])
    lines += _table("H1 weakest pruning candidates", sorted(h1, key=lambda row: (row.max_activation > EPS, row.activation_score, row.max_activation))[:20])
    lines += _table("H2 strongest active neurons", sorted(h2, key=lambda row: row.activation_score, reverse=True)[:20])
    lines += _table("H2 weakest pruning candidates", sorted(h2, key=lambda row: (row.max_activation > EPS, row.activation_score, row.max_activation))[:20])
    lines += ["", "## Scenario Scores", "", "| scenario | episodes | mean steps | mean score |", "| --- | ---: | ---: | ---: |"]
    for scenario in sorted({str(row["scenario"]) for row in episodes}):
        selected = [row for row in episodes if row["scenario"] == scenario]
        lines.append(f"| {scenario} | {len(selected)} | {np.mean([row['steps'] for row in selected]):.1f} | {np.mean([row['score'] for row in selected]):.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(title: str, rows: list[Metrics]) -> list[str]:
    lines = ["", f"## {title}", "", "| neuron | active freq | mean act | max act | outgoing abs-sum | activation score |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row.index} | {row.active_frequency:.4f} | {row.mean_activation:.6f} | {row.max_activation:.6f} | {row.outgoing_abs_sum:.3f} | {row.activation_score:.6f} |")
    return lines


def _draw_ranked(path: Path, h1: list[Metrics], h2: list[Metrics]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    for ax, rows, title in zip(axes, [h1, h2], ["H1", "H2"], strict=True):
        values = np.array(sorted((row.activation_score for row in rows), reverse=True))
        ax.bar(np.arange(len(values)), values, width=0.9, color="#5a82b8")
        ax.set_title(f"{title} activation-aware importance")
        ax.set_xlabel("rank")
        ax.set_ylabel("mean activation * outgoing abs-sum")
        ax.set_yscale("symlog", linthresh=1e-4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_frequency_histograms(path: Path, h1: list[Metrics], h2: list[Metrics]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    for ax, rows, title in zip(axes, [h1, h2], ["H1", "H2"], strict=True):
        ax.hist([row.active_frequency for row in rows], bins=25, color="#5a82b8", edgecolor="white")
        ax.set_title(f"{title} activation frequency")
        ax.set_xlabel("fraction of frames with activation > 0")
        ax.set_ylabel("neuron count")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
