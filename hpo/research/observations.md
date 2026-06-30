# HPO Observations

## 2026-06-30

**Observation:** In a 9D five-world SSL study with weighted Earth/Venus sampling, Optuna found a candidate around `176.4` Gym score in study `s1_go_optuna_go`.

**Context:** The run used `num_episodes=2000` and let Optuna search several HPs at once instead of manually tuning one or two parameters.

**HPs:** `learning_rate=0.0007590875386138993`, `batch_size=512`, `eps_end=0.03458570628736725`, `eps_decay=32323`, `gamma=0.995`, `tau=0.002`, `learning_starts=5000`, `optimize_every=2`, `replay_memory_capacity=40628`, `num_episodes=2000`.

**Interpretation:** The result is close to the historical preserved five-world `180` pilot and shows that wide Optuna search plus Earth/Venus flight hours can recover strong five-world behavior. The combination suggests a useful region: small fresh replay buffer, Earth-like epsilon corridor, moderate learning rate, longer-horizon `gamma`, slower target update `tau`, and later learning start.

**Next:** Let the study continue, then compare whether `gamma=0.995`, `tau=0.002`, and `learning_starts=5000` remain common among the best candidates.

## 2026-06-29

**Observation:** ==Breakthrough on Earth: 9D SSL reaches 200+.== In `solar_system_lander_9d_earth.db`, study `s7_exploration` found multiple candidates above `200` Gym score, with the best optimize trial around `242`. The best checkpoint that was actually preserved from the run is currently the robustness checkpoint around `206`.

![HP robustness evaluation for the 9D Earth breakthrough](assets/Durchbruch%20auf%20der%20Erde.png)

**Interpretation:** Earth is learnable for the small DQN. The observed `242` shows what is possible, but losing that concrete checkpoint shows that checkpoint preservation is now a core HPO requirement, not a convenience feature.

==**Next:** Implement automatic best-checkpoint preservation to Drive, then implement BI11 and try five-world training with world-weighted sampling for Earth/Venus.==

## 2026-06-28

**Observation:** The 8D Elise-bunt study produced a preserved five-world pilot around `180` Gym score.

**Interpretation:** This was an important historical milestone. It showed that a small shared SolarSystemLander for the inner worlds is plausible and gave confidence that the HPO path is worth pursuing, even though later 9D Earth-only results became the stronger current signal.

**Next:** Treat early strong individual models as directional evidence, but preserve their checkpoints immediately and keep separating concrete model quality from HP quality.

## YYYY-MM-DD

**Observation:** 

**Interpretation:** 

**Next:** 

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
