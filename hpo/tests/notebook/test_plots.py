import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from hpo.notebook import plots


def _scores() -> pd.DataFrame:
    return pd.DataFrame({
        "world": ["earth", "earth", "mars", "mars", "venus", "venus"],
        "score": [100.0, 200.0, 250.0, 300.0, -50.0, 150.0],
    })


def test_heatmap_returns_figure_and_axes_with_requested_world_order() -> None:
    fig, ax = plots.heatmap(_scores(), worlds=["venus", "earth"], bins=5)

    assert fig is ax.figure
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["venus", "earth"]
    plt.close(fig)


def test_histogram_3d_draws_in_reverse_world_order(monkeypatch) -> None:
    drawn_worlds = []

    original_add_subplot = plt.Figure.add_subplot

    def add_subplot(self, *args, **kwargs):
        ax = original_add_subplot(self, *args, **kwargs)

        def bar3d(_x, y, *_args, **bar_kwargs):
            drawn_worlds.append((bar_kwargs["label"], y))

        monkeypatch.setattr(ax, "bar3d", bar3d)
        return ax

    monkeypatch.setattr(plt.Figure, "add_subplot", add_subplot)
    monkeypatch.setattr(plt, "tight_layout", lambda: None)

    fig, ax = plots.histogram_3d(_scores(), worlds=["earth", "mars", "venus"], bins=5)

    assert fig is ax.figure
    assert drawn_worlds == [("venus", 2), ("mars", 1), ("earth", 0)]
    plt.close(fig)


def test_histogram_3d_uses_adaptive_score_tick_step() -> None:
    scores = pd.DataFrame({
        "world": ["earth", "earth"],
        "score": [-200.0, 350.0],
    })

    fig, ax = plots.histogram_3d(scores, bins=5)

    locator = ax.xaxis.get_major_locator()
    assert isinstance(locator, MultipleLocator)
    assert locator._edge.step == 100
    assert isinstance(ax.zaxis.get_major_locator(), MaxNLocator)
    assert ax.zaxis.get_major_locator()._integer
    plt.close(fig)


def test_quantiles_returns_figure_and_axes_with_requested_world_order() -> None:
    fig, ax = plots.quantiles(_scores(), worlds=["mars", "earth"])

    assert fig is ax.figure
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["mars", "earth"]
    plt.close(fig)
