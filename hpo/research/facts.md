# Facts

## VectorTrainer Throughput

**Context:** Empirical measurements from 103 trials on a Colab L4 with 20 parallel environments.

**Finding:** `optimize_every` explains nearly all observed throughput variation. Larger values mean fewer optimizer updates and therefore more environment steps per second.

**Formula:**

\[
\text{Env-Steps/s} \approx 851
\left(\frac{\text{optimize\_every}}{2}\right)^{0.854}
\]

| `optimize_every` | Measured Median Env-Steps/s |
|---:|---:|
| 2 | 858 |
| 4 | 1.483 |
| 8 | 2.666 |

**Cost model:** An environment step costs about `0.15 ms`; backpropagation plus optimizer update costs about `2.03 ms`. One optimizer update therefore costs about 14 environment steps.

**Interpretation:** The full four-HP regression reached about `1.1 %` mean relative error, but `batch_size`, `num_episodes`, and `learning_starts` added little explanatory value. The simple `optimize_every` formula is preferable.
