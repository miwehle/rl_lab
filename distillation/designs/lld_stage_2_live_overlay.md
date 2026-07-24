# LLD Stage 2 - Live Overlay

Goal: render the NN overlay dynamically during the landing video, using averaged per-step network state, while keeping the notebook thin and the first live version easy to inspect.

## Scope

Stage 2 extends `nn_viz.video.record_network_overlay_video(...)`.

It should:

- keep the Stage 1.5 trace and CSV output,
- keep the step label in the video,
- update the NN overlay during flight,
- smooth activations so the overlay does not flicker,
- reuse the existing `NetworkLayout` positions and static edge selection.

It should not:

- implement action RGB mixing,
- implement freeze/storyboard rendering,
- write exact per-step PNGs,
- add a separate video notebook mode unless the current notebook becomes confusing.

## Data Flow

For each policy decision step:

```text
observation_step
-> forward q_net
-> h1, h2, q_values
-> action = argmax(q_values)
-> append trace row
-> update averaged live state
-> env.step(action)
-> render frame with live overlay
```

The trace row, video label, live overlay, and executed action must all refer to the same decision step.

## Smoothing

Use a rolling mean for the moving video.

Initial KISS default:

```text
live_window_steps = 100
```

Maintain one rolling window per displayed signal:

```text
input_abs: [10]
h1: [hidden1]
h2: [hidden2]
q_values: [4]
```

The first steps use a growing initial window: at step 42 with `live_window_steps = 100`, the live overlay uses the mean of the 43 values collected so far. After the window is full, it uses the most recent `live_window_steps` values.

Use raw values for the trace. Use rolling means only for live rendering.

## Node Rendering

Reuse `NetworkLayout.nodes` for positions and base labels.

Node brightness:

- input nodes: normalized rolling mean of `abs(observation_i)`,
- H1/H2 nodes: normalized rolling mean of ReLU activations,
- output nodes: normalized rolling mean of relative Q-values.

For input/H1/H2, prefer fixed per-layer scales from the layout rollouts instead of per-frame layer maxima. The notebook computes `LIVE_NODE_SCALES` automatically from p95 values and passes them to `record_network_overlay_video(...)`. It also prints `max` beside p95 as a quick outlier check. If a scale is missing or zero, rendering falls back to the current frame's layer maximum.

Keep node size constant. Do not encode the same quantity twice via both size and brightness.

Output nodes may additionally show the currently chosen action with a simple outline or stronger fill. Keep this action highlight separate from action colors.

## Edge Rendering

Reuse `NetworkLayout.edges`; do not add more edges in Stage 2.

Edge sign stays as in the static plot:

- positive weight: green,
- negative weight: red.

Edge alpha/brightness is dynamic:

```text
edge_signal = source_signal * abs(weight)
```

where `source_signal` is the averaged input magnitude or hidden activation of the edge source node.

Keep edge thickness based on static `abs(weight)` for now. This avoids too much motion in the video.

## Rendering

Add a small live-rendering path in `nn_viz.video`, reusing the same bottom-overlay composition.

Suggested internal API:

```python
render_live_layout_rgba(layout, live_state, *, width, height, node_scales=None) -> np.ndarray
```

The existing static `render_layout_rgba(...)` remains useful for Stage 1 and for fallback.

`StaticNetworkOverlayWrapper` can become a more general overlay wrapper that accepts a callable returning the current RGBA overlay. Keep the public API stable unless a rename clearly makes the code simpler.

## Performance

Start by rendering the overlay for every environment step.

If this is too slow, add one simple lever later:

```text
refresh_every_n_steps
```

Between refreshes, reuse the last rendered overlay. Do not add this lever before a real slowdown is observed.

## Tests

Add focused pytest coverage for the non-visual mechanics:

- rolling mean uses a growing initial window and then only the most recent window,
- live state uses raw trace values only through the averager,
- edge signals use `source_signal * abs(weight)`,
- the wrapper can render with a changing overlay without changing frame shape,
- Stage 1.5 trace files are still written.

Do not attempt to test visual beauty in pytest. Use a generated preview frame or short video for manual inspection.
