# Distillation Observations

| Nr | Observation | Topics |
| --- | --- | --- |
| [[#O1 Micro-Elise 2 Is The Best Practical Checkpoint\|O1]] | Micro-Elise 2 Is The Best Practical Checkpoint | SSL, Distillation, Checkpointing |
| [[#O2 H1-16 Acts As Relative Up Disinhibition\|O2]] | H1-16 Acts As Relative Up Disinhibition | SSL, Distillation, Interpretability |
| [[#O3 Velocity Deltas Are Used But Not Score-Critical In Greedy Inference\|O3]] | Velocity Deltas Are Used But Not Score-Critical In Greedy Inference | SSL, Distillation, Interpretability |

Topics: `SSL` = SolarSystemLander.

## O1 Micro-Elise 2 Is The Best Practical Checkpoint

**Observation:** Among the current `64x64` Micro-Elise student checkpoints, run #2 is the best practical checkpoint: its mean score is essentially tied with #1, while its lower-tail robustness is better.

**When:** 2026-07-23

**Evidence:** Distillation run artifacts are in `G:\Meine Ablage\rl_lab\distillation\runs`. Each student was evaluated greedily with `100` episodes per world, so `500` episodes total per checkpoint. The full checkpoint comparison is in [[_details/O1|O1 details]].

The Elise-264-GSTP teacher still has a much stronger lower tail. Its `Q05` was recomputed with the same `evaluate_teacher(..., eval_episodes_per_world=100)` protocol used for the students.

| Modell | Mean | Q05 | Min | Median |
| --- | ---: | ---: | ---: | ---: |
| Elise-264-GSTP | 260.3 | **137.9** | -188.7 | 276.0 |
| Micro-Elise #2 | 248.7 | 39.7 | -211.0 | 269.9 |

**Interpretation:** Run #1 wins the mean by only `0.2` points, which is not enough to matter practically. Run #2 keeps essentially the same mean while improving the lower tail (`Q05 39.7` instead of `31.6`) and worst case (`-211` instead of `-324`), so it is the better current default checkpoint.

**Next:** Treat run #2, `solar_system_lander_10d_elise_stp_student_64x64_20260721T161456Z`, as the current Micro-Elise best checkpoint. The remaining gap is no longer median flight quality; it is lower-tail robustness, especially hard Venus/Earth cases.

## O2 H1-16 Acts As Relative Up Disinhibition

**Observation:** In Micro-Elise #2, hidden neuron `H1-16` behaves like a broad damping signal whose activity increases in heavier worlds and whose net effect makes `up` relatively more competitive.

**When:** 2026-07-23

**Evidence:** `H1-16` receives strong first-layer weights from vertical dynamics (`y_vel`, `dv_y`) and rotation. Ablating `H1-16` raises all four Q-values, but raises `noop`, `left`, and `right` more than `up`; therefore the neuron favors `up` only indirectly, by damping the other actions more. Details are in [[_details/O2|O2 details]].

| World | Gravity | Mean H1-16 | Up Share | Up Share Without H1-16 |
| --- | ---: | ---: | ---: | ---: |
| Moon | -1.65 | 0.453 | 0.072 | 0.002 |
| Mercury | -3.70 | 0.496 | 0.176 | 0.001 |
| Mars | -3.80 | 0.498 | 0.192 | 0.002 |
| Earth | -10.00 | 0.533 | 0.477 | 0.022 |
| Venus | -9.00 | 0.537 | 0.374 | 0.028 |

**Interpretation:** This is not a direct gravity-to-main-engine feature. In the `10d` observation mode the network sees gravity only indirectly through flight dynamics, especially velocity deltas. `H1-16` appears to use those signals as part of a learned ranking trick: it suppresses action values broadly while leaving `up` least suppressed, which can strongly increase main-engine selection in Earth/Venus.

## O3 Velocity Deltas Are Used But Not Score-Critical In Greedy Inference

**Observation:** Runtime ablation of the `10d` velocity-delta inputs (`dv_x`, `dv_y`) changes both Micro-Elise #2 and Elise-264-GSTP decisions, but it does not strongly reduce greedy evaluation score in the finished policies.

**When:** 2026-07-23

**Evidence:** The `nn_viz` notebook evaluated `normal`, `dv_x=0`, `dv_y=0`, and `dv_x+dv_y=0` with `100` greedy episodes per world for Micro-Elise #2 and Elise-264-GSTP. The full tables are in [[_details/O3|O3 details]].

| Model | Ablation | Mean Delta vs Normal | Mean Action Agreement |
| --- | --- | ---: | ---: |
| Micro-Elise #2 | `dv_x+dv_y=0` | -1.9 | 0.937 |
| Elise-264-GSTP | `dv_x+dv_y=0` | +0.2 | 0.956 |

**Interpretation:** `dv_x/dv_y` are not ignored: action agreement drops noticeably, especially when both are zeroed. But the score effect is small and mixed. This supports a distinction between training-time value and inference-time causal necessity: `dv_x/dv_y` likely helped the original HPO search find a good 10D pilot, while the finished greedy policies can often fly using other state cues.
