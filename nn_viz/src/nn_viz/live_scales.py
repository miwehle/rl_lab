"""Compute fixed scales for live NN video rendering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nn_viz.activations import ActivationRollouts
from nn_viz.layout import INPUT_LABELS, NetworkLayout


def compute_live_scales(
    rollouts: ActivationRollouts,
    layout: NetworkLayout,
    percentile: float = 95,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute fixed percentile scales for live video rendering."""
    percentile_label = f"p{percentile:g}"
    input_abs = np.abs(rollouts.observations)
    input_scales = np.percentile(input_abs, percentile, axis=0).astype(float)
    output_abs = np.abs(rollouts.q_values)
    hidden_values = np.concatenate([rollouts.h1.ravel(), rollouts.h2.ravel()])
    activation_values = np.concatenate([input_abs.ravel(), hidden_values])
    weights = np.array([abs(edge.weight) for edge in layout.edges], dtype=float)
    scales = {
        "input": input_scales,
        "hidden": float(np.percentile(hidden_values, percentile)),
        "output": float(np.percentile(output_abs, percentile)),
        "activation": float(np.percentile(activation_values, percentile)),
        "weight": float(np.percentile(weights, percentile)) if weights.size else 1.0,
    }
    rows = []
    for index, scale in enumerate(input_scales):
        rows.append(
            {
                "scope": f"input:{INPUT_LABELS[index]}",
                "max": float(np.max(input_abs[:, index])),
                percentile_label: float(scale),
            }
        )
    rows.extend(
        [
            {"scope": "hidden", "max": float(np.max(hidden_values)), percentile_label: scales["hidden"]},
            {"scope": "output", "max": float(np.max(output_abs)), percentile_label: scales["output"]},
            {
                "scope": "activation",
                "max": float(np.max(activation_values)),
                percentile_label: scales["activation"],
            },
            {
                "scope": "weight",
                "max": float(np.max(weights)) if weights.size else 0.0,
                percentile_label: scales["weight"],
            },
        ]
    )
    return scales, pd.DataFrame(rows)
