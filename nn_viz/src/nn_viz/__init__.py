"""Activation-based visualization helpers for Elise-like DQN networks."""

from nn_viz.activations import (
    ACTION_LABELS,
    ACTION_ORDER,
    ActivationRollouts,
    RolloutSpec,
    collect_activations,
    load_student_network,
)
from nn_viz.ablation import (
    DEFAULT_INPUT_ABLATIONS,
    InputAblation,
    evaluate_input_ablations,
)
from nn_viz.layout import (
    Edge,
    NetworkLayout,
    Node,
    compute_activity_layout,
    compute_semantic_layout,
)
from nn_viz.live_scales import compute_live_scales
from nn_viz.plot import plot_network_layout
from nn_viz.video import (
    LiveOverlayAverager,
    LiveOverlayState,
    StaticNetworkOverlayWrapper,
    compose_bottom_overlay,
    load_trace_state,
    record_network_overlay_video,
    render_live_layout_rgba,
    render_layout_rgba,
    render_trace_step_png,
)

__all__ = [
    "ACTION_LABELS",
    "ACTION_ORDER",
    "ActivationRollouts",
    "DEFAULT_INPUT_ABLATIONS",
    "Edge",
    "InputAblation",
    "LiveOverlayAverager",
    "LiveOverlayState",
    "NetworkLayout",
    "Node",
    "RolloutSpec",
    "StaticNetworkOverlayWrapper",
    "compose_bottom_overlay",
    "collect_activations",
    "compute_activity_layout",
    "compute_live_scales",
    "compute_semantic_layout",
    "evaluate_input_ablations",
    "load_student_network",
    "load_trace_state",
    "plot_network_layout",
    "record_network_overlay_video",
    "render_live_layout_rgba",
    "render_layout_rgba",
    "render_trace_step_png",
]
