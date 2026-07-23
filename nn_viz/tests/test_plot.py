import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nn_viz.layout import Edge, NetworkLayout, Node
from nn_viz.plot import plot_network_layout


def test_plot_network_layout_returns_figure():
    layout = NetworkLayout(
        nodes=(
            Node("out", 1, "left", 0.0, 0.0, 1.0),
            Node("h2", 0, "H2-0", 0.0, 1.0, 0.8),
            Node("h1", 0, "H1-0", 0.0, 2.0, 0.4),
        ),
        edges=(
            Edge("h2", 0, "out", 1, 0.5, 0.8, 0.7),
            Edge("h1", 0, "h2", 0, -0.4, 0.5, 0.5),
        ),
    )

    fig = plot_network_layout(layout)

    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_plot_places_hidden_layers_equidistant_with_constant_node_size():
    h2_nodes = tuple(Node("h2", index, f"H2-{index}", float(index * 2), 1.0, 0.2) for index in range(8))
    h1_nodes = tuple(Node("h1", index, f"H1-{index}", float(index * 2 - 3), 2.0, 0.1) for index in range(8))
    layout = NetworkLayout(
        nodes=(
            Node("out", 0, "left", 0.0, 0.0, 1.0),
            Node("out", 1, "up", 1.0, 0.0, 1.0),
            Node("out", 2, "noop", 2.0, 0.0, 1.0),
            Node("out", 3, "right", 3.0, 0.0, 1.0),
            *h2_nodes,
            *h1_nodes,
        ),
        edges=(
            Edge("h2", 0, "out", 0, 0.5, 0.8, 0.7),
            Edge("h1", 0, "h2", 0, -0.4, 0.5, 0.5),
        ),
    )

    fig = plot_network_layout(layout)

    try:
        h1_offsets = np.asarray(fig.axes[0].collections[0].get_offsets())
        h2_offsets = np.asarray(fig.axes[0].collections[1].get_offsets())
        output_offsets = np.asarray(fig.axes[0].collections[2].get_offsets())
        h1_sizes = fig.axes[0].collections[0].get_sizes()
        h2_sizes = fig.axes[0].collections[1].get_sizes()

        h1_spacing = np.diff(h1_offsets[:, 0])
        h2_spacing = np.diff(h2_offsets[:, 0])

        assert np.allclose(h1_spacing, 1.0)
        assert np.allclose(h2_spacing, 1.0)
        assert np.allclose(h1_offsets[:, 0], h2_offsets[:, 0])
        assert np.allclose(output_offsets[:, 0], [4.0, 6.0, 8.0, 10.0])
        assert len(set(h1_sizes)) == 1
        assert len(set(h2_sizes)) == 1
    finally:
        plt.close(fig)


def test_plot_labels_every_hidden_neuron_with_index():
    layout = NetworkLayout(
        nodes=(
            Node("out", 0, "left", 0.0, 0.0, 1.0),
            Node("out", 1, "up", 1.0, 0.0, 1.0),
            Node("h2", 0, "H2-0", 0.0, 1.0, 0.2),
            Node("h2", 9, "H2-9", 1.0, 1.0, 0.5),
            Node("h2", 10, "H2-10", 2.0, 1.0, 0.9),
            Node("h1", 0, "H1-0", 0.0, 2.0, 0.1),
            Node("h1", 10, "H1-10", 1.0, 2.0, 0.8),
            Node("h1", 11, "H1-11", 2.0, 2.0, 0.3),
        ),
        edges=(Edge("h2", 9, "out", 0, 0.5, 0.8, 0.7),),
    )

    fig = plot_network_layout(layout)

    try:
        labels = [text.get_text() for text in fig.axes[0].texts]
        assert labels == ["left", "up", "0", "9", "10", "0", "10", "11"]
    finally:
        plt.close(fig)
