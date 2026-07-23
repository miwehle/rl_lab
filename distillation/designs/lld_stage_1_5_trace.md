# LLD Stage 1.5 - Trace

Goal: record the NN state for each video step and show the current step in the video, without changing the static NN overlay into a live overlay yet.

## Scope

Stage 1.5 extends `nn_viz.video.record_network_overlay_video(...)`.

It should:

- keep the current static bottom NN overlay,
- display the current environment step in the video frame,
- save a per-step trace beside the video,
- keep notebooks thin.

It should not:

- implement live node or edge coloring,
- implement action colors,
- implement freeze/storyboard rendering,
- refactor HPO video recording.

## Trace Data

Write one trace file per recorded video, preferably next to the MP4 with the same stem:

```text
earth_seed_0_nn_overlay.mp4
earth_seed_0_nn_overlay_trace.npz
earth_seed_0_nn_overlay_trace_summary.csv
```

The trace contains one row per policy decision step:

```text
observations: [steps, 10]
actions: [steps]
h1: [steps, hidden1]
h2: [steps, hidden2]
q_values: [steps, 4]
```

For now, do not store edge inputs. They can be computed later from observations, activations, model weights, and the existing `NetworkLayout`.

The NPZ file is the machine-readable source of truth. Additionally write a small human-readable CSV summary with one row per step:

```text
step,action,q_left,q_up,q_noop,q_right
```

This is not meant to replace the NPZ. It is only for quick inspection while comparing a paused video frame with the policy decision at the same step.

## Step Synchronization

The trace row for `step = n` must correspond to the observation used to choose the action displayed for that step.

Loop order:

```text
reset -> observation_0
for step in range(max_steps):
    forward observation_step -> h1, h2, q_values
    action = argmax(q_values)
    append trace row for step
    env.step(action)
```

This matches the existing `collect_activations(...)` convention.

## Step Display

The video should draw a small `step: N` label on the rendered frame. It can be placed in a corner or near the existing score overlay; exact placement can be adjusted after visual inspection.

Implementation option: the `StaticNetworkOverlayWrapper` stores the last step number and adds the label inside `render()`. Since this is only simple text, using PIL or Pygame for the label is acceptable. Keep it local to `nn_viz.video`.

## Tests

Add focused pytest coverage for the trace mechanics, not for subjective video appearance.

Test invariants:

- one trace row per policy decision step,
- stored actions equal `argmax(q_values)`,
- array shapes match the model and observation space,
- CSV summary has the same step count as the NPZ trace,
- step numbering starts at `0` and is contiguous.

## Later Live Rendering

Stage 2 will use the trace-like values during recording to render a smooth live overlay.

Recommended live semantics:

- hidden/input nodes: brightness from smoothed activation/input magnitude,
- output nodes: brightness from smoothed relative Q-values plus current action highlight,
- edges: hue from weight sign and alpha/brightness from smoothed edge signal magnitude.

For moving video, use a rolling mean or EMA, e.g. `0.5 s`, not raw per-step values. Raw values are for exact step PNGs or freeze/storyboard frames.
