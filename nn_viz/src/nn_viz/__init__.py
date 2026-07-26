"""Activation-based visualization helpers for Elise-like DQN networks."""

from nn_viz.activations import (
    RolloutSpec,
    collect_activations,
    load_student_network,
)
from nn_viz.ablation import DEFAULT_INPUT_ABLATIONS, evaluate_input_ablations
from nn_viz.layout import compute_activity_layout, compute_semantic_layout
from nn_viz.scales import compute_scales
from nn_viz.trace import render_trace_diff, render_trace_step, render_trace_step_3d, render_trace_step_3d_html
from nn_viz.video import record_video

__all__ = [
    "DEFAULT_INPUT_ABLATIONS",
    "RolloutSpec",
    "collect_activations",
    "compute_activity_layout",
    "compute_scales",
    "compute_semantic_layout",
    "evaluate_input_ablations",
    "load_student_network",
    "record_video",
    "render_trace_diff",
    "render_trace_step",
    "render_trace_step_3d",
    "render_trace_step_3d_html",
]
