# Distillation Observations

| Nr | Observation | Topics |
| --- | --- | --- |
| [[#O1 Micro-Elise 2 Is The Best Practical Checkpoint\|O1]] | Micro-Elise 2 Is The Best Practical Checkpoint | SSL, Distillation, Checkpointing |

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
