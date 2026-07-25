"""Activation-based visualization helpers for Elise-like DQN networks."""

from nn_viz.activations import (
    RolloutSpec,
    collect_activations,
    load_student_network,
)
from nn_viz.ablation import DEFAULT_INPUT_ABLATIONS, evaluate_input_ablations
from nn_viz.layout import compute_activity_layout, compute_semantic_layout
from nn_viz.live_scales import compute_live_scales
from nn_viz.video import record_video, render_trace_diff_png, render_trace_step_png

__all__ = [
    "DEFAULT_INPUT_ABLATIONS",
    "RolloutSpec",
    "collect_activations",
    "compute_activity_layout",
    "compute_live_scales",
    "compute_semantic_layout",
    "evaluate_input_ablations",
    "load_student_network",
    "record_video",
    "render_trace_diff_png",
    "render_trace_step_png",
]
