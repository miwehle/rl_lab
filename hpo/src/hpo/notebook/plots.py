from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, MultipleLocator


def heatmap(
    scores: pd.DataFrame,
    worlds: Sequence[str] | None = None,
    *,
    bins: int = 30,
):
    import matplotlib.pyplot as plt

    worlds = _worlds(scores, worlds)
    bin_edges = _bin_edges(scores, bins)
    y = np.arange(len(worlds))
    grouped_scores = scores.groupby("world")["score"]
    histograms = [
        np.histogram(_scores_for_world(scores, world), bins=bin_edges)[0]
        for world in worlds
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(
        histograms,
        aspect="auto",
        origin="lower",
        extent=[bin_edges[0], bin_edges[-1], -0.5, len(worlds) - 0.5],
    )

    ax.scatter(
        grouped_scores.median().reindex(worlds),
        y,
        marker="o",
        s=70,
        facecolors="white",
        edgecolors="black",
        label="median",
    )
    ax.scatter(
        grouped_scores.mean().reindex(worlds),
        y,
        marker="x",
        s=70,
        color="tab:red",
        linewidths=2,
        label="mean",
    )

    ax.set(xlabel="score", ylabel="world")
    ax.set_yticks(y, worlds)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    fig.colorbar(image, ax=ax, label="episodes")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, borderaxespad=0)
    plt.tight_layout()
    return fig, ax


def histogram_3d(
    scores: pd.DataFrame,
    worlds: Sequence[str] | None = None,
    *,
    bins: int = 30,
    reverse_draw_order: bool = True,
):
    import matplotlib.pyplot as plt

    worlds = _worlds(scores, worlds)
    draw_worlds = list(reversed(worlds)) if reverse_draw_order else worlds
    bin_edges = _bin_edges(scores, bins)
    left_edges = bin_edges[:-1]
    width = bin_edges[1] - bin_edges[0]

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)

    for world in draw_worlds:
        y = worlds.index(world)
        counts, _ = np.histogram(_scores_for_world(scores, world), bins=bin_edges)
        ax.bar3d(
            left_edges,
            y,
            0,
            width * 0.9,
            0.45,
            counts,
            alpha=0.6,
            zsort="max",
            label=world,
        )

    ax.set(xlabel="score", ylabel="world", zlabel="episodes")
    ax.set_yticks(range(len(worlds)), worlds)
    ax.xaxis.set_major_locator(MultipleLocator(_tick_step(scores["score"])))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_zlabel("episodes", labelpad=4)
    ax.view_init(elev=24, azim=-45)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.05, top=0.98)
    return fig, ax


def quantiles(scores: pd.DataFrame, worlds: Sequence[str] | None = None):
    import matplotlib.pyplot as plt

    worlds = _worlds(scores, worlds)
    summary = _summary(scores).reindex(worlds)
    y = np.arange(len(worlds))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hlines(
        y,
        summary["min"],
        summary["max"],
        color="0.80",
        linewidth=2,
        label="min..max",
    )
    ax.hlines(
        y,
        summary["q05"],
        summary["q95"],
        color="tab:blue",
        linewidth=5,
        alpha=0.45,
        label="q05..q95",
    )
    ax.hlines(
        y,
        summary["q25"],
        summary["q75"],
        color="tab:blue",
        linewidth=12,
        alpha=0.85,
        label="q25..q75",
    )
    ax.scatter(
        summary["median"],
        y,
        color="white",
        edgecolor="black",
        zorder=3,
        label="median",
    )
    ax.scatter(
        summary["mean"],
        y,
        marker="x",
        color="tab:red",
        zorder=3,
        label="mean",
    )

    ax.set(xlabel="score", ylabel="world")
    ax.set_yticks(y, worlds)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=5,
        borderaxespad=0,
    )
    plt.tight_layout()
    return fig, ax


def _worlds(scores: pd.DataFrame, worlds: Sequence[str] | None) -> list[str]:
    available = set(scores["world"].dropna())
    if worlds is None:
        return scores["world"].dropna().drop_duplicates().tolist()
    return [world for world in worlds if world in available]


def _scores_for_world(scores: pd.DataFrame, world: str) -> pd.Series:
    return scores.loc[scores["world"] == world, "score"].dropna()


def _bin_edges(scores: pd.DataFrame, bins: int) -> np.ndarray:
    values = scores["score"].dropna()
    score_min = float(values.min())
    score_max = float(values.max())
    if score_min == score_max:
        score_min -= 1.0
        score_max += 1.0
    return np.linspace(score_min, score_max, bins + 1)


def _tick_step(values: pd.Series, target_ticks: int = 6) -> float:
    values = values.dropna()
    span = float(values.max() - values.min())
    if span <= 0:
        return 1.0

    raw_step = span / target_ticks
    magnitude = 10 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude

    if normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10

    return float(nice * magnitude)


def _summary(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.groupby("world")["score"].agg(
        mean="mean",
        min="min",
        q05=lambda score: score.quantile(0.05),
        q25=lambda score: score.quantile(0.25),
        median="median",
        q75=lambda score: score.quantile(0.75),
        q95=lambda score: score.quantile(0.95),
        max="max",
    )
