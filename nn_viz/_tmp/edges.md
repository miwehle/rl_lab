# NN Viz Edges

This note explains how nn_viz chooses and renders network edges.

## Mental Model

A layout edge is a candidate connection between two neurons.

A rendered edge is a frame-specific visual object derived from:

- the layout edge weight
- the current source activation
- the current scales
- the selected renderer backend

## Static Layout Edges

`NetworkLayout.edges` stores the stable edge candidates.

Each `Edge` contains:

- source layer and source index
- target layer and target index
- weight
- relevance
- specificity

Layout builders usually select ==top-k edges per target from rollout statistics==.

## Dynamic Video Edges

Video rendering starts from all network weights, then selects visible edges per frame with `representative_edges(...)`.

For each target neuron, positive and negative contributors are split by current contribution:

```text
contribution = source_value * weight
```

The requested edge budget is divided between positive and negative contributors by their total absolute contribution.

## Edge Appearance

2D edges encode three things:

- sign: red/blue from signed contribution
- weight strength: color saturation and line width
- source activation: alpha

The color sign uses:

```text
color_value = sign(source_value * weight) * abs(weight)
```

So a negative weight can become visually red if a negative source activation makes the contribution positive.

## Draw Order

Visible edges are sorted before drawing.

Low visual priority edges are drawn first; stronger colored edges are drawn later so they remain visible.

```text
visual_priority = colorfulness * alpha * nominal_width
```

This mainly keeps pale near-white edges from covering important red/blue paths.

## Renderer Backends

The 2D renderer supports:

- `pillow`: default, simple
- `aggdraw`: smoother edges, slower

Both backends receive the same sorted drawable edge list.