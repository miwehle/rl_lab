# HPO Observations

| Nr                                                                    | Observation                                               | Topics             |
| --------------------------------------------------------------------- | --------------------------------------------------------- | ------------------ |
| [[#O11 Trial-Cluster Confirms 10D HP Corridor\|O11]]                  | Trial-Cluster Confirms 10D HP Corridor                   | SSL, HP, Optuna    |
| [[#O10 Five-World 10D Reaches 253\|O10]]                              | Five-World 10D Reaches 253                                | SSL, HP            |
| [[#O9 Colab Runtime Ended After 8.5 Hours\|O9]]                       | Colab Runtime Ended After 8.5 Hours                       | OTO, PERF          |
| [[#O8 Five-World 10D Reaches 210\|O8]]                                | Five-World 10D Reaches 210                                | SSL, HP            |
| [[#O7 Top HP Corridor Emerges\|O7]]                                   | Top HP Corridor Emerges                                   | SSL, HP, Optuna    |
| [[#O6 Five-World 9D Reaches 176\|O6]]                                 | Five-World 9D Reaches 176                                 | SSL, HP, Optuna    |
| [[#O5 Sampling Mix Shapes World Scores\|O5]]                          | Sampling Mix Shapes World Scores                          | SSL, Sampling      |
| [[#O4 Early Good HP Corridor In 9D\|O4]]                              | Early Good HP Corridor In 9D                              | SSL, HP            |
| [[#O3 Earth Breakthrough\|O3]]                                        | Earth Breakthrough                                        | SSL, Checkpointing |
| [[#O2 8D Elise-Bunt Produces 180 Pilot\|O2]]                          | 8D Elise-Bunt Produces 180 Pilot                          | SSL, LL            |
| [[#O1 VectorTrainer Throughput Depends Mostly On optimize_every\|O1]] | VectorTrainer Throughput Depends Mostly On optimize_every | PERF               |

Topics: `RL` = Reinforcement Learning, `SSL` = SolarSystemLander, `OTO` = Optimize the Optimizer, `LL` = Lessons Learned, `HP` = Hyperparameters, `PERF` = Performance/Throughput.

## O11 Trial-Cluster Confirms 10D HP Corridor

**Observation:** In `s3_10d_better_space`, trials `10..15` form a compact HP cluster, and every second trial in that cluster produced a good pilot: trial 10 scored `160.0`, trial 12 scored `193.5`, trial 13 scored `210.1`, and trial 15 scored `178.0`.

**When:** 2026-07-02.

**Evidence:** The cluster shares `gamma=0.995`, `tau=0.002`, `batch_size=512`, `learning_starts=2500`, `num_episodes=2000`, `eps_end≈0.040..0.044`, `eps_decay≈32k..47k`, and replay roughly `82k..116k`. The later 253 pilot in trial 35 is similar: `gamma=0.995`, `tau=0.002`, `eps_end=0.044`, `eps_decay=38793`, replay `76754`, and learning rate `0.000623`.

**Interpretation:** This is evidence that Optuna found a genuinely useful HP corridor. It is not a robust factory, because nearby trial 11 still scored only `7.6`; but it is a better pilot lottery, with clearly elevated probability of producing quality checkpoints.

## O10 Five-World 10D Reaches 253

**Observation:** In `solar_system_lander_10d_elise_accel.db`, study `s3_10d_better_space`, 10D reached `252.6` average Gym score over five worlds in trial 35.

**When:**
- 2026-07-01 morning: 253 pilot observed.
- 2026-07-02: extended with 9D-vs-10D search-budget evidence.

**Context:** This used the 10D acceleration observation mode and the refined HP region. The checkpoint was preserved in Drive as `best_checkpoints/solar_system_lander_10d_elise_accel/best_eval_checkpoint.pt`.

**World scores:** Earth `218.8`, Mars `281.2`, Mercury `266.7`, Moon `256.9`, Venus `239.3`.

**Evidence:** 9D had more search budget without producing such a pilot: about `93` trials with `2000` episodes each, and the best observed 9D pilot reached only about `204`. 10D had about `45` trials with `2000` episodes each, and the `253` pilot appeared in trial 35.

**Interpretation:** ==10D is now the leading SSL path. The 253 pilot scores above `200` on every world, so this is not a single-world spike but a true five-world pilot.==

**Details:** [[_details/O10|HPs]]

## O9 Colab Runtime Ended After 8.5 Hours

**Observation:** Colab ended the 10D SSL study cell after about `8 h 30 min 33 s` of execution. The notebook showed `Wieder verbinden`, and the study had not reached its target yet.

**When:** 2026-07-01 05:25, after `30633 s` runtime.

**Evidence:** Screenshot: [Beweisfoto Runtime Crash](assets/Beweisfoto%20Runtime%20Crash.png).

**Related observation:** Two trials with extremely poor Gym score (< 1280) at transfer from the training plot into the study plot appeared near Colab runtime-loss events. These trials were not resource-heavy compared with normal trials, so resource load is unlikely as the direct cause. The temporal coincidence remains suspicious but unexplained.

**Interpretation:** This does not prove the exact cause, but it records a concrete Colab reliability event. Repeated entries may reveal whether Colab tends to interrupt long runs around certain wall-clock times or durations.

## O8 Five-World 10D Reaches 210

**Observation:** In `solar_system_lander_10d_elise_accel.db`, study `s3_10d_better_space`, 10D reached about `210` average Gym score over five worlds.

**When:** 2026-07-01.

**Context:** This used the 10D acceleration observation mode and a narrowed HP region after 9D/10D exploration.

**Interpretation:** 10D is a serious candidate, not just an experiment. The best observed 10D pilot is currently slightly ahead of the best observed 9D five-world pilot.

**Details:** [[_details/O7|O7]]

## O7 Top HP Corridor Emerges

**Observation:** Top models currently share a clear HP core:

**When:** 2026-07-01.

```text
gamma: 0.995
tau: 0.002

lr: ~0.001
eps_end: 0.03..0.04
eps_decay: 30k..50k; ~100k can work with eps_end ~0.04

replay: unclear
  9D: 20k..50k
  10D: ~85k
```

**Interpretation:** `gamma` and `tau` are now strong fixed candidates. Replay size remains interaction-heavy and deserves a more focused follow-up.

**Details:** [[_details/O7|O7]]

## O6 Five-World 9D Reaches 176

**Observation:** In a 9D five-world SSL study with weighted Earth/Venus sampling, Optuna found a candidate around `176.4` Gym score in study `s1_go_optuna_go`.

**When:** 2026-07-01.

**Context:** The run used `num_episodes=2000` and let Optuna search several HPs at once instead of manually tuning one or two parameters.

**HPs:** `learning_rate=0.0007590875386138993`, `batch_size=512`, `eps_end=0.03458570628736725`, `eps_decay=32323`, `gamma=0.995`, `tau=0.002`, `learning_starts=5000`, `optimize_every=2`, `replay_memory_capacity=40628`, `num_episodes=2000`.

**Interpretation:** The result is close to the historical preserved five-world `180` pilot and shows that wide Optuna search plus Earth/Venus flight hours can recover strong five-world behavior. The combination suggests a useful region: small fresh replay buffer, Earth-like epsilon corridor, moderate learning rate, longer-horizon `gamma`, slower target update `tau`, and later learning start.

**Next:** Let the study continue, then compare whether `gamma=0.995`, `tau=0.002`, and `learning_starts=5000` remain common among the best candidates.

## O5 Sampling Mix Shapes World Scores

**Observation:** In the same 9D five-world study, the sampling mix was `Mercury 1x, Venus 4x, Earth 4x, Moon 1x, Mars 1x`. After 27 trials with `world_scores`, Mars and Mercury were usually the strongest worlds, while Moon, Earth, and Venus most often pulled the mean down.

**When:** 2026-07-01.

**Evidence:** Mean world scores were roughly `Mars 124.5`, `Mercury 116.0`, `Moon 36.4`, `Earth 8.1`, `Venus 7.3`. The weakest world per trial was Venus 9 times, Moon 9 times, Earth 8 times, and Mercury once.

**Interpretation:** Mars appears close to the middle of the five-world flight regime and is therefore easiest for the shared SSL. ==Earth/Venus still need more own flight hours; Moon also needs more preservation== because tuning toward high-g/weather regimes can hurt low-g landing quality.

**Next:** Try a slightly stronger static sampling mix such as `Mercury 1x, Mars 1x, Moon 2x, Earth 5x, Venus 5x` and use `num_envs=28`.

## O4 Early Good HP Corridor In 9D

**Observation:** In the 9D Go-Optuna-Go DB around trial 26, good early values looked roughly like this:

**When:** 2026-07-01.

```text
learning_rate:            0.00058 .. 0.00080
eps_end:                  0.029 .. 0.044
eps_decay:                27_776 .. 60_585
replay_memory_capacity:   28_968 .. 96_222
learning_starts:          mostly 5_000
optimize_every:           mostly 2
gamma:                    0.99 or 0.995, best trial 0.995
tau:                      0.002 or 0.01, best trial 0.002
num_episodes:             2000
batch_size:               512
```

The leading trial at that point was:

```text
score: 176.4
learning_rate: 0.000759
eps_end: 0.034586
eps_decay: 32323
gamma: 0.995
tau: 0.002
learning_starts: 5000
optimize_every: 2
replay_memory_capacity: 40628
num_episodes: 2000
batch_size: 512
```

**Interpretation:** These ranges came from top trials, not guessing. Later results refined the corridor, especially around `gamma=0.995` and `tau=0.002`.

## O3 Earth Breakthrough

**Observation:** ==Breakthrough on Earth: 9D SSL reaches 200+.== In `solar_system_lander_9d_earth.db`, study `s7_exploration` found multiple candidates above `200` Gym score, with the best optimize trial around `242`. The best checkpoint that was actually preserved from the run is currently the robustness checkpoint around `206`.

**When:** 2026-06-29.

![HP robustness evaluation for the 9D Earth breakthrough](assets/Durchbruch%20auf%20der%20Erde.png)

**Interpretation:** Earth is learnable for the small DQN. The observed `242` shows what is possible, but losing that concrete checkpoint shows that checkpoint preservation is now a core HPO requirement, not a convenience feature.

==**Next:** Implement automatic best-checkpoint preservation to Drive, then implement BI11 and try five-world training with world-weighted sampling for Earth/Venus.==

## O2 8D Elise-Bunt Produces 180 Pilot

**Observation:** The 8D Elise-bunt study produced a preserved five-world pilot around `180` Gym score.

**When:** 2026-06-29.

**Interpretation:** This was an important historical milestone. It showed that a small shared SolarSystemLander for the inner worlds is plausible and gave confidence that the HPO path is worth pursuing, even though later 9D Earth-only results became the stronger current signal.

**Next:** Treat early strong individual models as directional evidence, but preserve their checkpoints immediately and keep separating concrete model quality from HP quality.

## O1 VectorTrainer Throughput Depends Mostly On optimize_every

**Observation:** Empirical measurements from 103 trials on a Colab L4 with 20 parallel environments.

| `optimize_every` | Measured Median Env-Steps/s |
| ---------------: | --------------------------: |
|                2 |                         858 |
|                4 |                       1.483 |
|                8 |                       2.666 |

**When:** 2026-06-29.

**Finding:** `optimize_every` explains nearly all observed throughput variation. Larger values mean fewer optimizer updates and therefore more environment steps per second.

**Formula:**

\[
\text{Env-Steps/s} \approx 851
\left(\frac{\text{optimize\_every}}{2}\right)^{0.854}
\]

**Cost model:** An environment step costs about `0.15 ms`; backpropagation plus optimizer update costs about `2.03 ms`. One optimizer update therefore costs about 14 environment steps.

**Interpretation:** The full four-HP regression reached about `1.1 %` mean relative error, but `batch_size`, `num_episodes`, and `learning_starts` added little explanatory value. The simple `optimize_every` formula is preferable.
