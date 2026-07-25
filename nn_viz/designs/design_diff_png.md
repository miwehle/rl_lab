# NN Diff PNG

## Goal

Compare two backward-window mean NN states from the same trace and render the difference as a standalone PNG.

Public function:

```python
render_trace_diff(
    trace_path,
    layout,
    output_path,
    *,
    from_step,
    to_step,
    from_window_steps=1,
    to_window_steps=1,
    width=1280,
    height=360,
)
```

The diff direction is:

```text
to_step/to_window - from_step/from_window
```

This should help identify which neurons and displayed edges change between flight phases.

## State Diff

For each side, compute the same backward-window mean state used by `render_trace_step(...)`.

```text
inputs_delta = inputs_to - inputs_from
h1_delta = h1_to - h1_from
h2_delta = h2_to - h2_from
q_delta = q_to - q_from
edge_delta = (source_to - source_from) * weight
```

`window_steps=1` means a raw single step. Larger windows mean a backward mean ending at the selected step. At the beginning of a trace, the window grows from step 0.

## Color Semantics

Use rollout-based scales computed internally from the saved trace, not diff-local scales. This avoids exaggerating tiny changes and keeps the public API small.

Inputs use component-wise scales:

```text
scales = compute scales from the full trace
signed_color(input_delta_i, scales["input"][i])
```

Hidden neurons use one shared hidden scale for H1 and H2:

```text
signed_color(hidden_delta, scales["hidden"])
```

This is the main difference from normal hidden rendering: normal hidden activations use `heat_color(...)` because ReLU activations are nonnegative; hidden diffs use `signed_color(...)` because the difference can be negative.

Outputs use the output scale:

```text
signed_color(q_delta, scales["output"])
```

Displayed edges use contribution deltas:

```text
edge_delta = (source_to - source_from) * weight
edge_scale = scales["activation"] * scales["weight"]
signed_color(edge_delta, edge_scale)
alpha(edge_delta, edge_scale)
```

Edge width should still use `edge_width(weight, scales["weight"])`, so visually important weights do not disappear just because their current diff is small.

## Labels

Use index labels by default. Diff PNGs are analysis artifacts, not video overlays.

## Implementation Shape

Expose `render_trace_diff(...)` as a separate public function, but do not duplicate the full state renderer.

Refactor the existing state rendering path only as much as needed so normal rendering and diff rendering share the same private layout/canvas/node/edge/label machinery. The diff renderer should provide different node and edge color rules, not a copied renderer.

## Notebook

Add a compact `render-diff-png` example cell to `nn_viz/notebooks/micro_elise_nn_video.ipynb` after `render-step-png`.

The cell should set `DIFF_FROM_STEP`, `DIFF_TO_STEP`, `DIFF_FROM_WINDOW_STEPS`, and `DIFF_TO_WINDOW_STEPS`, then call `render_trace_diff(...)` for the current `trace_path` and `layout`.

## Non-Goals

Do not build a video diff overlay yet.

Do not compare different traces/checkpoints yet.

Do not add special input normalization beyond the existing per-input rollout scales.

