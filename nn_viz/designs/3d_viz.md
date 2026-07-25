# 3D NN Visualization

## Goal

Add an experimental 3D visualization for Elise-like networks without disturbing the existing 2D video/trace renderer.

The purpose is exploratory: find semantic structure in NN spaghetti by making learned connections and activations visible in space.

## Design

`NetworkLayout` remains the data model for node and edge geometry. `Node` gains a `z` coordinate; existing 2D layouts set `z = 0.0`, while a new 3D layout can use meaningful depth.

The render engine stays separate from the layout. A layout answers "where are the nodes and edges"; a renderer answers "how is that shown". Therefore, do not bind `engine="pyvista"` or similar settings to `NetworkLayout`.

## Modules

```text
nn_viz/layout/...
  existing 2D layouts set z=0.0

nn_viz/layout/spatial/semantic.py
  compute_layout(...)

nn_viz/_pyvista_rendering.py
  internal PyVista renderer for 3D snapshots/scenes
```

The first renderer module should be explicitly PyVista-specific. A generic `spatial_rendering.py` layer is unnecessary until there is more than one 3D rendering backend.

## First Step

Implement the smallest useful snapshot path:

```python
render_layout_snapshot(layout, output_path, ...)
```

It should first draw nodes as spheres and edges as simple thin lines, using `node.x`, `node.y`, and `node.z`. Tubes, labels, styling, and animation can wait.

The first useful trace-facing helper should follow soon after:

```python
render_trace_step_3d(trace_path, layout, output_path, *, step, window_steps=1, ...)
```

It should load the saved trace state like `nn_viz.trace.render_trace_step(...)`, then render a real PyVista 3D scene as a screenshot.

3D layouts are intended for PyVista rendering. Existing 2D renderers may ignore `z` for compatibility, but rendering a 3D layout as a flat 2D projection is not a goal.

Visual semantics should match the 2D renderer where practical:

- node color/brightness follows activation
- edge thickness follows absolute weight
- edge color follows weight sign
- edge activation display is still open

For 3D/PyVista, edge geometry and edge styling should be treated separately. Creating tubes/lines may be the expensive part; once an edge actor exists, changing its color or opacity may be cheap enough. Therefore, do not assume yet that source activation should be implemented by removing edges per frame. A static edge set with dynamic color/opacity updates may be the better runtime model.

Edge signal semantics:

```text
edge width     = abs(w)
edge color     = sign(x * w)
edge intensity = abs(x)
```

For hidden-layer edges, `x >= 0` because of ReLU, so `sign(x * w)` is equivalent to `sign(w)`. For input edges, `x` can be negative, so `sign(x * w)` shows the actual current contribution.

`edge intensity` should be render-mode dependent:

- saturation/brightness mode: probably faster and more robust
- opacity/alpha mode: probably visually nicer, like colored liquid flowing through tubes

The PyVista renderer should make this switchable so both modes can be compared.

## Implementation Stages

Stage 1: Data model and compatibility.

- add `Node.z: float = 0.0`
- keep existing 2D rendering behavior unchanged
- update tests and CSV handling only where needed

Stage 2: Spatial semantic layout.

- add `nn_viz/layout/spatial/semantic.py`
- implement `compute_layout(...)`
- test fixed input/output anchors
- test H1/H2 weighted-mean placement
- test layer heights via `z`

Stage 2.5: PyVista smoke snapshot.

- add `nn_viz/_pyvista_rendering.py`
- implement `render_layout_snapshot(layout, output_path, ...)`
- offscreen PNG only
- render nodes as simple spheres
- render edges as simple thin lines
- use fixed neutral colors
- no tubes
- no labels
- no trace-state coloring

Stage 3: Styled PyVista snapshot.

- add edge thickness from `abs(w)`, likely via tubes
- add node color from activity or state
- add edge color/intensity modes
- keep labels and dynamic coloring optional for later

Stage 4: Trace-step 3D screenshot.

- implement `render_trace_step_3d(trace_path, layout, output_path, *, step, window_steps=1, ...)`
- reuse saved trace state loading
- render one real rollout step as a PyVista screenshot

Stage 5: Dynamic 3D trace playback.

- reuse fixed 3D geometry
- add rollout-synchronized coloring of nodes and edges
- use lower frame rate or sampled steps if needed

## Later: Dynamic 3D Trace Playback

A dynamic 3D NN should be added after the static trace-step screenshot works.

The intended runtime model:

- build node and edge geometry once
- keep edge thickness static because it represents `abs(w)`
- update node colors from activations
- update edge color/intensity from `x`, `w`, and `x * w`
- use a lower frame rate or sampled steps if needed

Rendering at the full 2D video rate is not required for analysis. A rate such as `5-10 fps`, or explicit step sampling, is acceptable if it keeps the 3D scene responsive and readable.

## First 3D Layout

Use `z` as layer height in the 3D scene:

```text
z = 3  output
z = 2  hidden 2
z = 1  hidden 1
z = 0  input
```

Input and output get fixed semantic anchor coordinates in the `x/y` plane. Hidden layers are placed dynamically between those anchors.

Input anchors:

```text
ftl   ftr

y     vy    ay

x     vx    ax

ang   vang
```

Output anchors:

```text
left  right

up    noop
```

The left/right semantics of input and output should roughly share the same `x/y` coordinates and differ mainly by height. For example, `ftl` should sit roughly below `left`, and `ftr` roughly below `right`.

Output placement is fixed by these anchors. H2 nodes are connected to the output layer according to the underlying NN weights `w3`; those edges do not move the output anchors.

Implementation module:

```text
nn_viz/layout/spatial/semantic.py
  compute_layout(...)
```

The short function name is intentional: the package and module already say that this is the spatial semantic layout.

H1 placement:

```text
h1_xy = weighted mean of input anchor xy positions
weight(input i) = abs(w1[h1, i])
```

This keeps mixed H1 neurons between their relevant input anchors instead of snapping them to only the single strongest input. If all weights are zero, use the center of the input anchors as stable fallback.

H2 placement:

```text
h2_xy = weighted mean of H1 xy positions
weight(h1 i) = abs(w2[h2, i])
```

This mirrors H1 placement one layer higher. If all weights are zero, use the center of the H1 positions as stable fallback.

Hidden collision handling:

Weighted means tend to pull hidden nodes toward the center, especially in H2. To keep the 3D scene readable without drifting whole layers away from their semantic positions, treat the weighted-mean result as the desired position and then resolve close node collisions inside each hidden layer.

```python
compute_layout(..., min_node_distance=0.14)
```

The collision pass moves only hidden node `x/y` positions. It keeps the hidden layer center fixed, leaves `z` unchanged, and must not affect edge selection, weights, activations, or trace values.

Hidden neurons resist collision displacement according to incoming weight strength:

```text
stiffness = sum(abs(incoming_weights)) + eps
mobility  = 1 / stiffness
```

During overlap removal, a less stiff neuron moves more than a stiffer one. This keeps strongly anchored neurons closer to their semantic weighted-mean position while weaker neurons make room.

Edge selection:

Use all weights for node placement, but do not render all nonzero weights as 3D tubes. A Micro-Elise `10-64-64-4` network has up to `4992` possible edges, which is too much visual and rendering noise for a first useful 3D view.

The spatial layout should therefore select rendered edges with Top-k per target neuron, analogous to the 2D layouts:

```python
compute_layout(..., top_edges_per_target=3, output_edges_per_target=10)
```

- H1/H2 positions still use all absolute incoming weights.
- Input->H1 and H1->H2 rendered edges use `top_edges_per_target`.
- H2->Output rendered edges use `output_edges_per_target`.
- Top-k ranking is by `abs(weight)` for the static 3D layout.

## Open Questions

- Where should the optional PyVista dependency be documented or declared?

## Notebook Integration

3D snapshot cells should live in the existing `nn_viz/notebooks/micro_elise_nn_video.ipynb`.

Reason: the video notebook already creates the relevant context: `layout`, `trace_path`, selected step, window size, and result directory. Keeping the 3D cells there avoids opening a second notebook just to inspect one rollout step in 3D.

First notebook cell:

```text
# cell: render-3d-smoke; requires: build-layout
```

Later, after `render_trace_step_3d(...)` exists:

```text
# cell: render-trace-step-3d; requires: record-video
```
