# LLD Stage 2 - Live Overlay

Goal: render the NN overlay dynamically during the landing video, using smoothed per-step network state, while keeping the notebook thin and the first live version easy to inspect.

## Scope

Stage 2 extends `nn_viz.video.record_network_overlay_video(...)`.

It should:

- keep the Stage 1.5 trace and CSV output,
- keep the step/action label in the video,
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
-> update smoothed live state
-> env.step(action)
-> render frame with live overlay
```

The trace row, video label, live overlay, and executed action must all refer to the same decision step.

## Smoothing

Use an EMA for the moving video.

Initial KISS default:

```text
ema_alpha = 0.15
```

Maintain one EMA state per displayed signal:

```text
input_abs: [10]
h1: [hidden1]
h2: [hidden2]
q_values: [4]
```

Use raw values for the trace. Use EMA values only for live rendering.

## Node Rendering

Reuse `NetworkLayout.nodes` for positions and base labels.

Node brightness:

- input nodes: normalized EMA of `abs(observation_i)`,
- H1/H2 nodes: normalized EMA of ReLU activations,
- output nodes: normalized EMA of relative Q-values.

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

where `source_signal` is the smoothed input magnitude or hidden activation of the edge source node.

Keep edge thickness based on static `abs(weight)` for now. This avoids too much motion in the video.

## Rendering

Add a small live-rendering path in `nn_viz.video`, reusing the same bottom-overlay composition.

Suggested internal API:

```python
render_live_layout_rgba(layout, live_state, *, width, height) -> np.ndarray
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

- EMA updates converge toward new values,
- live state uses raw trace values only through the smoother,
- edge signals use `source_signal * abs(weight)`,
- the wrapper can render with a changing overlay without changing frame shape,
- Stage 1.5 trace files are still written.

Do not attempt to test visual beauty in pytest. Use a generated preview frame or short video for manual inspection.
