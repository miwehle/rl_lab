"""Render hidden-neuron importance distributions for the Elise checkpoint."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

CHECKPOINT_DIR = Path(r"G:\Meine Ablage\rl_lab\hpo\best_checkpoints\solar_system_lander_10d_elise_stp")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_eval_checkpoint.pt"
OUT_DIR = Path(__file__).resolve().parent
POPOMETER_H1 = 80


def main() -> int:
    state = _load_state_dict(CHECKPOINT_PATH)
    layer1 = state["layer1.weight"].cpu().numpy()
    layer2 = state["layer2.weight"].cpu().numpy()
    layer3 = state["layer3.weight"].cpu().numpy()
    h1_importance = np.abs(layer1).sum(axis=1) * np.abs(layer2).sum(axis=0)
    h2_importance = np.abs(layer2).sum(axis=1) * np.abs(layer3).sum(axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _draw_importance_distributions(h1_importance, h2_importance, OUT_DIR / "elise_neuron_importance_distributions.png")
    _draw_ranked_importance(h1_importance, h2_importance, OUT_DIR / "elise_neuron_importance_ranked.png")
    _write_table(h1_importance, h2_importance, OUT_DIR / "elise_neuron_importance_top20.md")
    print(f"wrote neuron-importance visualizations to: {OUT_DIR}")
    return 0


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint


def _draw_importance_distributions(h1: np.ndarray, h2: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    for ax, values, title in zip(axes, [h1, h2], ["hidden 1", "hidden 2"], strict=True):
        ax.hist(values, bins=30, color="#5a82b8", edgecolor="white")
        ax.axvline(np.median(values), color="#333333", linestyle="--", linewidth=1.1, label="median")
        ax.axvline(np.quantile(values, 0.9), color="#b85a32", linestyle="--", linewidth=1.1, label="90% quantile")
        ax.set_title(f"{title} importance distribution")
        ax.set_xlabel("incoming abs-sum * outgoing abs-sum")
        ax.set_ylabel("neuron count")
        ax.legend()
    fig.suptitle("Elise hidden-neuron importance distributions", y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_ranked_importance(h1: np.ndarray, h2: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    for ax, values, title in zip(axes, [h1, h2], ["hidden 1", "hidden 2"], strict=True):
        order = np.argsort(values)[::-1]
        ranked = values[order]
        colors = ["#f0c419" if title == "hidden 1" and neuron == POPOMETER_H1 else "#5a82b8" for neuron in order]
        ax.bar(np.arange(len(ranked)), ranked, color=colors, width=0.9)
        ax.set_title(f"{title} ranked importance")
        ax.set_xlabel("rank")
        ax.set_ylabel("importance")
        ax.set_yscale("log")
        if title == "hidden 1":
            rank = int(np.where(order == POPOMETER_H1)[0][0])
            ax.annotate("H1-80", xy=(rank, ranked[rank]), xytext=(rank + 6, ranked[rank] * 0.55), arrowprops={"arrowstyle": "->", "color": "#5a4500"}, color="#5a4500")
    fig.suptitle("Elise hidden-neuron importance by rank", y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(h1: np.ndarray, h2: np.ndarray, path: Path) -> None:
    lines = ["# Elise Neuron Importance Top 20", ""]
    for name, values in [("Hidden 1", h1), ("Hidden 2", h2)]:
        lines += [f"## {name}", "", "| rank | neuron | importance |", "| ---: | ---: | ---: |"]
        for rank, neuron in enumerate(np.argsort(values)[::-1][:20], start=1):
            lines.append(f"| {rank} | {int(neuron)} | {values[neuron]:.3f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
