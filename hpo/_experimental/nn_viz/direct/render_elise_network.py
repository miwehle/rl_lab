"""Render network visualizations for the Elise best checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib.collections import LineCollection

CHECKPOINT_DIR = Path(r"G:\Meine Ablage\rl_lab\hpo\best_checkpoints\solar_system_lander_10d_elise_stp")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_eval_checkpoint.pt"
METADATA_PATH = CHECKPOINT_DIR / "best_eval_checkpoint.json"

OUT_DIR = Path(__file__).resolve().parent
INPUT_LABELS = [
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular velocity",
    "left leg",
    "right leg",
    "dv_x",
    "dv_y",
]
ACTION_LABELS = ["noop", "left", "main", "right"]


def main() -> int:
    state = _load_state_dict(CHECKPOINT_PATH)
    weights = [
        state["layer1.weight"].cpu().numpy(),
        state["layer2.weight"].cpu().numpy(),
        state["layer3.weight"].cpu().numpy(),
    ]
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _draw_topk_network(weights, OUT_DIR / "elise_topk_network.png", OUT_DIR / "elise_topk_network.svg", top_k=3)
    _draw_weight_heatmaps(weights, OUT_DIR / "elise_weight_heatmaps.png")
    _draw_interactive_network(weights, OUT_DIR / "elise_interactive_topk_network.html", top_k=5)
    _write_summary(metadata, weights, OUT_DIR / "README.md")
    print(f"wrote visualizations to: {OUT_DIR}")
    return 0


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint


def _draw_topk_network(weights: list[np.ndarray], png_path: Path, svg_path: Path, *, top_k: int) -> None:
    layers = [weights[0].shape[1], weights[0].shape[0], weights[1].shape[0], weights[2].shape[0]]
    positions = _layer_positions(layers)
    edges = _topk_edges(weights, top_k=top_k)
    max_abs = max(abs(weight) for *_nodes, weight in edges)

    fig, ax = plt.subplots(figsize=(18, 10), dpi=180)
    for sign, color in [(1, "#b83232"), (-1, "#2f65b0")]:
        segments = []
        widths = []
        for src, dst, weight in edges:
            if np.sign(weight) != sign:
                continue
            segments.append([positions[src], positions[dst]])
            strength = abs(weight) / max_abs
            widths.append(0.15 + 2.8 * strength)
        collection = LineCollection(segments, colors=color, linewidths=widths, alpha=0.35)
        ax.add_collection(collection)

    for layer_index, layer_size in enumerate(layers):
        xs = [positions[(layer_index, node)][0] for node in range(layer_size)]
        ys = [positions[(layer_index, node)][1] for node in range(layer_size)]
        node_size = 70 if layer_size <= 10 else 10
        ax.scatter(xs, ys, s=node_size, c="#f4f4f4", edgecolors="#222222", linewidths=0.4, zorder=3)

    _label_layers(ax, layers)
    ax.set_title("Elise DQN: all neurons, strongest incoming edges per neuron", pad=18)
    ax.set_axis_off()
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-1.06, 1.06)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close(fig)


def _draw_weight_heatmaps(weights: list[np.ndarray], path: Path) -> None:
    titles = ["layer1: input -> hidden 1", "layer2: hidden 1 -> hidden 2", "layer3: hidden 2 -> actions"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    for ax, matrix, title in zip(axes, weights, titles, strict=True):
        limit = np.quantile(np.abs(matrix), 0.995)
        image = ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xlabel("source neuron")
        ax.set_ylabel("target neuron")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Elise DQN weight matrices", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_interactive_network(weights: list[np.ndarray], path: Path, *, top_k: int) -> None:
    layers = [weights[0].shape[1], weights[0].shape[0], weights[1].shape[0], weights[2].shape[0]]
    positions = _layer_positions(layers)
    edges = _topk_edges(weights, top_k=top_k)
    max_abs = max(abs(weight) for *_nodes, weight in edges)

    fig = go.Figure()
    for src, dst, weight in edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        strength = abs(weight) / max_abs
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={
                    "color": f"rgba({184 if weight >= 0 else 47}, {50 if weight >= 0 else 101}, {50 if weight >= 0 else 176}, {0.12 + 0.65 * strength:.3f})",
                    "width": 0.2 + 4.0 * strength,
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    hover_x = []
    hover_y = []
    hover_text = []
    for src, dst, weight in edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        hover_x.append((x0 + x1) / 2)
        hover_y.append((y0 + y1) / 2)
        hover_text.append(f"{_node_label(src, layers)} -> {_node_label(dst, layers)}<br>weight={weight:.5f}")
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker={"size": 8, "color": "rgba(0,0,0,0)"},
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    for layer_index, layer_size in enumerate(layers):
        xs = [positions[(layer_index, node)][0] for node in range(layer_size)]
        ys = [positions[(layer_index, node)][1] for node in range(layer_size)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker={"size": 11 if layer_size <= 10 else 4, "color": "#f4f4f4", "line": {"color": "#222", "width": 0.8}},
                hovertext=[_node_label((layer_index, node), layers) for node in range(layer_size)],
                hoverinfo="text",
                name=["input", "hidden 1", "hidden 2", "actions"][layer_index],
            )
        )

    fig.update_layout(
        title=f"Elise DQN interactive top-{top_k} incoming-edge network",
        width=1400,
        height=900,
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        showlegend=True,
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _topk_edges(weights: list[np.ndarray], *, top_k: int) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    edges = []
    for layer_index, matrix in enumerate(weights):
        k = min(top_k, matrix.shape[1])
        for target in range(matrix.shape[0]):
            source_indexes = np.argpartition(np.abs(matrix[target]), -k)[-k:]
            for source in source_indexes:
                edges.append(((layer_index, int(source)), (layer_index + 1, target), float(matrix[target, source])))
    return edges


def _layer_positions(layers: list[int]) -> dict[tuple[int, int], tuple[float, float]]:
    positions = {}
    for layer_index, layer_size in enumerate(layers):
        ys = np.linspace(0.95, -0.95, layer_size) if layer_size > 1 else np.array([0.0])
        for node, y in enumerate(ys):
            positions[(layer_index, node)] = (float(layer_index), float(y))
    return positions


def _label_layers(ax, layers: list[int]) -> None:
    names = ["input", "hidden 1", "hidden 2", "actions"]
    for layer_index, (name, size) in enumerate(zip(names, layers, strict=True)):
        ax.text(layer_index, 1.02, f"{name}\n{size}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for index, label in enumerate(INPUT_LABELS):
        ax.text(-0.06, _layer_positions(layers)[(0, index)][1], label, ha="right", va="center", fontsize=7)
    for index, label in enumerate(ACTION_LABELS):
        ax.text(3.06, _layer_positions(layers)[(3, index)][1], label, ha="left", va="center", fontsize=9)


def _node_label(node: tuple[int, int], layers: list[int]) -> str:
    layer, index = node
    if layer == 0:
        return f"input {index}: {INPUT_LABELS[index]}"
    if layer == 3:
        return f"action {index}: {ACTION_LABELS[index]}"
    return f"hidden {layer} neuron {index}"


def _write_summary(metadata: dict, weights: list[np.ndarray], path: Path) -> None:
    score = metadata.get("score")
    hidden_size = metadata.get("training_config", {}).get("hidden_size")
    edge_count = sum(matrix.size for matrix in weights)
    path.write_text(
        "\n".join(
            [
                "# Elise Network Visualizations",
                "",
                f"Checkpoint: `{CHECKPOINT_PATH}`",
                f"Score in metadata: `{score}`",
                f"Hidden size: `{hidden_size}`",
                f"Total weights: `{edge_count}`",
                "",
                "Generated files:",
                "",
                "- `elise_topk_network.png` / `elise_topk_network.svg`: all neurons, strongest incoming edges per neuron.",
                "- `elise_weight_heatmaps.png`: complete weight matrices.",
                "- `elise_interactive_topk_network.html`: zoomable/hoverable top-k edge graph.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
