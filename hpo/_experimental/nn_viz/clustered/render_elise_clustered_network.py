"""Render clustered network visualizations for the Elise best checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

CHECKPOINT_DIR = Path(r"G:\Meine Ablage\rl_lab\hpo\best_checkpoints\solar_system_lander_10d_elise_stp")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_eval_checkpoint.pt"
METADATA_PATH = CHECKPOINT_DIR / "best_eval_checkpoint.json"

OUT_DIR = Path(__file__).resolve().parent
POPOMETER_H1 = 80
ACTION_LABELS = ["noop", "left", "main", "right"]


def main() -> int:
    state = _load_state_dict(CHECKPOINT_PATH)
    weights = [
        state["layer1.weight"].cpu().numpy(),
        state["layer2.weight"].cpu().numpy(),
        state["layer3.weight"].cpu().numpy(),
    ]
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    h2_by_h1 = weights[1]
    h2_order = _cluster_order(np.abs(h2_by_h1))
    h1_order = _cluster_order(np.abs(h2_by_h1).T)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _draw_clustered_layer2_heatmap(h2_by_h1, h2_order, h1_order, OUT_DIR / "elise_clustered_layer2_heatmap.png")
    _draw_clustered_bipartite(h2_by_h1, h2_order, h1_order, OUT_DIR / "elise_clustered_bipartite_top_edges.png", OUT_DIR / "elise_clustered_bipartite_top_edges.svg")
    _draw_top_neuron_cluster(weights, OUT_DIR / "elise_clustered_top_neurons.png", OUT_DIR / "elise_clustered_top_neurons.html")
    _write_summary(metadata, weights, OUT_DIR / "README.md")
    print(f"wrote clustered visualizations to: {OUT_DIR}")
    return 0


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint


def _cluster_order(features: np.ndarray) -> np.ndarray:
    distances = pdist(features, metric="cosine")
    distances = np.nan_to_num(distances, nan=0.0)
    return leaves_list(linkage(distances, method="average"))


def _draw_clustered_layer2_heatmap(matrix: np.ndarray, h2_order: np.ndarray, h1_order: np.ndarray, path: Path) -> None:
    ordered = matrix[h2_order][:, h1_order]
    popometer_column = int(np.where(h1_order == POPOMETER_H1)[0][0])
    limit = np.quantile(np.abs(matrix), 0.995)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    image = ax.imshow(ordered, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    ax.axvline(popometer_column, color="#f0c419", linewidth=1.6)
    ax.text(popometer_column, -3, "H1-80", ha="center", va="bottom", fontsize=8, color="#5a4500")
    ax.set_title("Elise layer2 weights clustered by connection profile")
    ax.set_xlabel("hidden 1 neurons, clustered")
    ax.set_ylabel("hidden 2 neurons, clustered")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_clustered_bipartite(matrix: np.ndarray, h2_order: np.ndarray, h1_order: np.ndarray, png_path: Path, svg_path: Path) -> None:
    h1_rank = {int(neuron): rank for rank, neuron in enumerate(h1_order)}
    h2_rank = {int(neuron): rank for rank, neuron in enumerate(h2_order)}
    h1_y = _rank_positions(len(h1_order))
    h2_y = _rank_positions(len(h2_order))
    edges = _topk_layer_edges(matrix, top_k=4)
    max_abs = max(abs(weight) for *_nodes, weight in edges)

    fig, ax = plt.subplots(figsize=(10, 13), dpi=180)
    for sign, color in [(1, "#b83232"), (-1, "#2f65b0")]:
        segments = []
        widths = []
        for h1, h2, weight in edges:
            if np.sign(weight) != sign:
                continue
            strength = abs(weight) / max_abs
            segments.append([(0.0, h1_y[h1_rank[h1]]), (1.0, h2_y[h2_rank[h2]])])
            widths.append(0.12 + 2.7 * strength)
        ax.add_collection(LineCollection(segments, colors=color, linewidths=widths, alpha=0.32))

    ax.scatter(np.zeros(len(h1_order)), h1_y, s=9, c="#f4f4f4", edgecolors="#222", linewidths=0.3, zorder=3)
    ax.scatter(np.ones(len(h2_order)), h2_y, s=9, c="#f4f4f4", edgecolors="#222", linewidths=0.3, zorder=3)
    ax.scatter([0], [h1_y[h1_rank[POPOMETER_H1]]], s=60, c="#f0c419", edgecolors="#5a4500", linewidths=0.7, zorder=4)
    ax.text(-0.035, h1_y[h1_rank[POPOMETER_H1]], "H1-80", ha="right", va="center", fontsize=8, color="#5a4500")
    ax.text(0, 1.04, "hidden 1\nclustered", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(1, 1.04, "hidden 2\nclustered", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Elise clustered bipartite layout: strongest H1 -> H2 edges", pad=18)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-1.04, 1.04)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _draw_top_neuron_cluster(weights: list[np.ndarray], png_path: Path, html_path: Path) -> None:
    h2_by_h1 = weights[1]
    h1_importance = np.abs(h2_by_h1).sum(axis=0)
    h2_importance = np.abs(h2_by_h1).sum(axis=1) + np.abs(weights[2]).sum(axis=0)
    h1_keep = _ordered_top_indexes(h1_importance, count=30, force=POPOMETER_H1)
    h2_keep = _ordered_top_indexes(h2_importance, count=30)
    submatrix = h2_by_h1[np.ix_(h2_keep, h1_keep)]
    h2_order = _cluster_order(np.abs(submatrix))
    h1_order = _cluster_order(np.abs(submatrix).T)
    h1_keep = h1_keep[h1_order]
    h2_keep = h2_keep[h2_order]
    submatrix = h2_by_h1[np.ix_(h2_keep, h1_keep)]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    limit = np.quantile(np.abs(submatrix), 0.995)
    image = ax.imshow(submatrix, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_title("Elise top 30 H1/H2 neurons, clustered")
    ax.set_xlabel("hidden 1 top neurons")
    ax.set_ylabel("hidden 2 top neurons")
    ax.set_xticks(range(len(h1_keep)), [f"H1-{i}" for i in h1_keep], rotation=90, fontsize=6)
    ax.set_yticks(range(len(h2_keep)), [f"H2-{i}" for i in h2_keep], fontsize=6)
    if POPOMETER_H1 in h1_keep:
        ax.axvline(int(np.where(h1_keep == POPOMETER_H1)[0][0]), color="#f0c419", linewidth=1.5)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    _write_top_neuron_html(submatrix, h1_keep, h2_keep, html_path)


def _write_top_neuron_html(matrix: np.ndarray, h1_keep: np.ndarray, h2_keep: np.ndarray, path: Path) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"H1-{i}" for i in h1_keep],
            y=[f"H2-{i}" for i in h2_keep],
            colorscale="RdBu",
            zmid=0,
            hovertemplate="source=%{x}<br>target=%{y}<br>weight=%{z:.5f}<extra></extra>",
        )
    )
    fig.update_layout(title="Elise top 30 H1/H2 neurons, clustered", width=1000, height=900, template="plotly_white")
    fig.write_html(path, include_plotlyjs="cdn")


def _topk_layer_edges(matrix: np.ndarray, *, top_k: int) -> list[tuple[int, int, float]]:
    edges = []
    for h2 in range(matrix.shape[0]):
        h1_indexes = np.argpartition(np.abs(matrix[h2]), -top_k)[-top_k:]
        for h1 in h1_indexes:
            edges.append((int(h1), h2, float(matrix[h2, h1])))
    return edges


def _rank_positions(count: int) -> np.ndarray:
    return np.linspace(0.98, -0.98, count)


def _ordered_top_indexes(values: np.ndarray, *, count: int, force: int | None = None) -> np.ndarray:
    indexes = np.argsort(values)[::-1][:count]
    if force is not None and force not in indexes:
        indexes[-1] = force
    return np.array(sorted(set(int(index) for index in indexes)), dtype=int)


def _write_summary(metadata: dict, weights: list[np.ndarray], path: Path) -> None:
    score = metadata.get("score")
    hidden_size = metadata.get("training_config", {}).get("hidden_size")
    edge_count = sum(matrix.size for matrix in weights)
    path.write_text(
        "\n".join(
            [
                "# Elise Clustered Network Visualizations",
                "",
                f"Checkpoint: `{CHECKPOINT_PATH}`",
                f"Score in metadata: `{score}`",
                f"Hidden size: `{hidden_size}`",
                f"Total weights: `{edge_count}`",
                "",
                "Generated files:",
                "",
                "- `elise_clustered_layer2_heatmap.png`: complete `H2 x H1` layer2 matrix, rows and columns clustered by absolute connection profile.",
                "- `elise_clustered_bipartite_top_edges.png` / `.svg`: H1 and H2 reordered by cluster, with strongest incoming H1 edges per H2.",
                "- `elise_clustered_top_neurons.png` / `.html`: top 30 H1/H2 neurons by connection strength, clustered and labeled.",
                "",
                "`H1-80` is highlighted because it is the dominant dv/popometer-coupled hidden-1 neuron.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
