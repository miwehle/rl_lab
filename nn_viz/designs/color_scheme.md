# Color Scheme

## Scale

`scale` is a fixed reference value computed before rendering, usually as p95 over the relevant rollout values.

Examples:

- input scale for `x`: `p95(abs(x) over rollout steps)`,
- hidden scale for H2: `p95(H2 activations over rollout steps)`,
- output scale: `p95(abs(q values or relative q values) over rollout steps)`,
- weight scale: `p95(abs(weights) over displayed or all NN weights)`.

During rendering, the current step value is compared with this fixed scale. Values near `0` are weak; values near or above `scale` are strong. Values above `scale` are clipped.

## Core Functions

```python
alpha(value, scale) -> int
signed_color(value, scale) -> RGB
heat_color(value, scale) -> RGB
edge_width(weight, scale) -> float
```

`alpha(value, scale)` uses `abs(value)` and maps it to `0..255`. A logarithmic mapping is preferred:

```text
ratio = log1p(abs(value)) / log1p(scale)
alpha = alpha_min + ratio * (alpha_max - alpha_min)
```

`signed_color(value, scale)` uses the sign for hue and `abs(value) / scale` for color strength:

```text
value < 0: blue
value = 0: gray
value > 0: red
```

`heat_color(value, scale)` is for ReLU activations, so there is no sign. Use a heat-like scale with logarithmic compression.

`edge_width(weight, scale)` uses `abs(weight)` and returns a nominal width with logarithmic compression. It does not know the image size.

The renderer converts nominal widths to pixels for its target medium, e.g. by multiplying with an overlay-height factor, rounding, and clamping to a minimum visible width. This keeps render geometry out of the color scheme.

## Usage

Edges:

```python
a = alpha(source_activation, activation_scale)
color = signed_color(weight, weight_scale)
nominal_width = edge_width(weight, weight_scale)
```

Input neurons:

```python
a = alpha(input_value, input_scale)
color = signed_color(input_value, input_scale)
```

Hidden neurons:

```python
color = heat_color(activation, hidden_scale)
```

Output neurons:

```python
a = alpha(q_value, output_scale)
color = signed_color(q_value, output_scale)
```

## Open Decision

For output neurons, decide whether `q_value` means raw signed Q-value or relative action preference such as `q - mean(q)` or `q - min(q)`. For video readability, relative action preference is likely better, but signed raw Q-values may be more literal.
