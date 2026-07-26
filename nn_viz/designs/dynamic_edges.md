# Dynamic Edges

## Goal

Make visible NN edges depend on the current rendered state instead of on a fixed edge subset from the layout.

The layout should answer where neurons are placed. The renderer should answer which edges matter in the current state.

## Edge Contribution

For each rendered state, compute the current contribution of every possible edge:

```text
contribution(edge) = source_value * weight
```

Examples:

```text
Input -> H1:  input_i * w1[j, i]
H1 -> H2:     h1_i * w2[j, i]
H2 -> Output: h2_i * w3[j, i]
```

## Visibility

Visible edges should approximate the incoming weighted sum of each target neuron.

For every target neuron, select a small fixed number of incoming edges. Split that edge budget between positive and negative contributions according to their current share of the target's absolute incoming contribution:

```text
pos_sum = sum(contribution where contribution > 0)
neg_sum = sum(abs(contribution) where contribution < 0)
total = pos_sum + neg_sum

k_pos = round(edge_contributors_per_target * pos_sum / total)
k_neg = edge_contributors_per_target - k_pos

show the strongest k_pos positive edges
show the strongest k_neg negative edges
```

Use one parameter:

```python
edge_contributors_per_target=6
```

If `total == 0`, all incoming edge contributions for that target are zero in the current state, so no incoming edge is representative for that target.

Apply this independently per target in each edge layer:

```text
Input -> H1
H1 -> H2
H2 -> Output
```

This keeps rendering cost bounded while making visible connections more representative of the target activation than a global Top-k or quantile over `abs(x*w)`.

Bias terms are not part of edge visibility. Visible edges approximate the weighted input sum `sum(x*w)`, not `bias + sum(x*w)`. Bias should be visualized separately later, for example by a neuron outline, if it becomes important.

Do not force a minimum edge count per sign. If one sign contributes almost nothing, it may disappear from the visible edges.

## Rendering Semantics

For every visible edge:

```text
color hue   = sign(x * w)
color power = abs(w)
alpha       = abs(x)
width       = abs(w)
```

The same edge visibility and rendering semantics should be used by 2D video, 2D trace PNG, 3D trace PNG, and 3D trace HTML.

Trace files store the network weight matrices so trace renderers can reconstruct all possible edges without needing the original `q_net` object.

## Layout

Node positions may still be computed from rollout activity so the layout remains stable. Edge visibility is state-based and may change from step to step.
