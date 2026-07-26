# Dynamic Edges

## Goal

Make visible NN edges depend on the current rendered state instead of on a fixed edge subset from the layout.

The layout should answer where neurons are placed. The renderer should answer which edges matter in the current state.

## Edge Effect

For each rendered state, compute the current effect of every possible edge:

```text
effect(edge) = abs(source_value * weight)
```

Examples:

```text
Input -> H1:  abs(input_i * w1[j, i])
H1 -> H2:     abs(h1_i * w2[j, i])
H2 -> Output: abs(h2_i * w3[j, i])
```

## Visibility

Visible edges are selected per state by layerwise quantile over `abs(x*w)`.

This matches opacity semantics: edges that would be nearly invisible are not rendered.

Use one parameter:

```python
edge_effect_quantile=0.95
```

Apply it separately for:

```text
Input -> H1
H1 -> H2
H2 -> Output
```

Layerwise filtering prevents the large H1->H2 layer from hiding input and output effects.

## Rendering Semantics

For every visible edge:

```text
color hue   = sign(x * w)
color power = abs(w)
alpha       = abs(x)
width       = abs(w)
```

The same edge visibility and rendering semantics should be used by 2D video, 2D trace PNG, 3D trace PNG, and 3D trace HTML.

## Layout

Node positions may still be computed from rollout activity so the layout remains stable. Edge visibility is state-based and may change from step to step.
