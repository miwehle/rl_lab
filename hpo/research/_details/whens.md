# Research Item Dates

Purpose: Reconstructed dates for research items whose dates are not kept inline. Observation dates stay inline in `observations.md`.

| Item Nr. | Date | Change in short |
|---|---|---|
| [[../aha#A15 Artificial Armstrong Is A Narrow Domain Criterion\|A15]] | 2026-07-12 | Added the narrow Artificial Armstrong criterion for Elise-264-GSTP. |
| [[../aha#A14 Efficient Pilots Need Dynamic Safety Margin\|A14]] | 2026-07-12 | Recorded Elise-264-GSTP crash-analysis lesson from worst crash videos. |
| [[../aha#A13 Fine-Tune Strong Pilots Into A Model Line\|A13]] | 2026-07-11 | Recorded Elise-264-GSTP as a checkpoint fine-tuning/model-line lesson. |
| [[../hypotheses#H7 Strong Turbulence Can Saturate The Action Channel\|H7]] | 2026-07-12 | Added action-channel saturation hypothesis for worst Elise-264-GSTP crashes. |
| [[../aha#A1 Code Complexity Is Part Of The Experiment\|A1]] | 2026-06-29 | Recorded code complexity as part of the experiment. |
| [[../ideas#Begriffe aus dem Apollo-Programm\|ideas/Apollo]] | 2026-06-29 | Added Apollo-style terminology for checkpoints, campaigns, qualification, and flight testing. |
| [[../aha#A12 How To Build A Small Good Five-World Lander\|A12]] | 2026-07-01 | Added initial synthesis of how to build a small good five-world lander. |
| [[../aha#A11 Preserve Good Pilots Immediately\|A11]] | 2026-07-01 | Recorded the checkpoint-preservation lesson from lost and preserved pilots. |
| [[../aha#A10 10D Gives The SSL A Popometer\|A10]] | 2026-07-01 | Added the Popometer mental model for 10D acceleration observations. |
| [[../aha#A9 Earth Is Learnable\|A9]] | 2026-07-01 | Recorded that Earth is learnable with the right setup. |
| [[../aha#A8 Hard Worlds Need Their Own Flight Hours\|A8]] | 2026-07-01 | Recorded that hard worlds need more own training exposure. |
| [[../aha#A7 Good HPs Are Not Enough\|A7]] | 2026-07-01 | Recorded that HPs are producers, not concrete models. |
| [[../aha#A6 Observation Mode Is Not Settled\|A6]] | 2026-07-01 | Recorded that 8D, 9D, and 11D are not fairly settled yet. |
| [[../aha#A5 Visualize Early\|A5]] | 2026-07-01 | Recorded the dashboard as a diagnostic instrument. |
| [[../aha#A4 Let Optuna Explore\|A4]] | 2026-07-01 | Recorded the lesson to let Optuna search broadly when the situation is unclear. |
| [[../aha#A3 Gamma And Tau Shape Learning Dynamics\|A3]] | 2026-07-01 | Added the gamma/tau learning-dynamics explanation. |
| [[../aha#A2 Back Up Immediately\|A2]] | 2026-07-01 | Recorded that a good checkpoint counts only after preservation. |
| [[../hypotheses#H1 Earth Is Learnable\|H1]] | 2026-07-01 | Added Earth-learnability hypothesis. |
| [[../hypotheses#H2 Hard Worlds Need Flight Hours\|H2]] | 2026-07-01 | Added hard-world flight-hours hypothesis. |
| [[../hypotheses#H3 Sampling Should Favor Hard Worlds\|H3]] | 2026-07-01 | Added world-dependent sampling hypothesis. |
| [[../hypotheses#H4 Observation Mode Is Still Open\|H4]] | 2026-07-01 | Added open observation-mode hypothesis. |
| [[../hypotheses#H5 Good HPs Are Not Enough\|H5]] | 2026-07-01 | Added hypothesis that good HPs do not guarantee a good model. |
| [[../questions#Q2 Training Nutzen Bei Epsilon Unter 0.05\|Q2]] | 2026-07-01 | Added question whether training below epsilon `0.05` still helps. |
| [[../questions#Q1 SolarSystemLander Difficulty\|Q1]] | 2026-07-01 | Added question what makes Earth and Venus hard. |
| [[../ideas#Pilot Preservation Ideas\|ideas/Pilot Preservation]] | 2026-06-30 | Added initial pilot-preservation ideas. |
| [[../ideas#Video vom Training\|ideas/Video]] | 2026-06-30 | Added video-recording idea. |
| [[../ideas#Dashboard liest aus DB\|ideas/Dashboard DB]] | 2026-06-30 | Added idea that dashboard plots could read from DB. |
| [[../aha#A12 How To Build A Small Good Five-World Lander\|A12]] | 2026-07-02 | Extended after the 253 pilot and Popometer/dashboard synthesis. |
| [[../ideas#Pilot Preservation Ideas\|ideas/Pilot Preservation]] | 2026-07-02 | Added early talent signal idea. |
| [[../ouch#E1 Selection Bias In HP Robustness\|E1]] | 2026-07-02 | Recorded HP robustness selection-bias error. |

## Hints For Detective Codex

Last checked Git commit: `66c1374`

Last checked at: `2026-07-02`

Scope:
- `aha.md`
- `hypotheses.md`
- `questions.md`
- `ideas.md`
- `ouch.md`

Future updates only need to inspect commits after `66c1374`, for example:

```powershell
git log 66c1374..HEAD -- hpo/research
```

Evidence method: dates were reconstructed with `git log -S "<item heading>"`, unless stated otherwise in `Change in short`.
