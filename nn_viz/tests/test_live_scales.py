import numpy as np

from nn_viz.activations import ActivationRollouts
from nn_viz.layout import Edge, NetworkLayout
from nn_viz.live_scales import compute_live_scales


def test_compute_live_scales_returns_scales_and_summary():
    rollouts = ActivationRollouts(
        observations=np.array([[1, -2, 3, -4, 5, -6, 0, 1, -8, 9]], dtype=np.float32),
        h1=np.array([[1, 2]], dtype=np.float32),
        h2=np.array([[3, 4]], dtype=np.float32),
        q_values=np.array([[-1, 2, -3, 4]], dtype=np.float32),
        actions=np.array([3], dtype=np.int64),
        rows=(),
    )
    layout = NetworkLayout(
        nodes=(),
        edges=(
            Edge("h1", 0, "h2", 0, 0.25, 0.0, 0.0),
            Edge("h2", 0, "out", 0, -2.0, 0.0, 0.0),
        ),
    )

    scales, summary = compute_live_scales(rollouts, layout, percentile=100)

    np.testing.assert_array_equal(scales["input"], [1, 2, 3, 4, 5, 6, 0, 1, 8, 9])
    assert scales["hidden"] == 4.0
    assert scales["output"] == 4.0
    assert scales["activation"] == 9.0
    assert scales["weight"] == 2.0
    assert summary["scope"].tolist()[:2] == ["input:x", "input:y"]
