# Research Item Dates

Purpose: Investigation state for research item dates. Main files stay readable; this file keeps the reconstructed timeline for items where the date is not part of the research content itself.

Observation dates stay inline in `observations.md`, because time is usually part of the observation.

## Investigation State

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

| Item | Date | Evidence |
|---|---|---|
| A12 | 2026-07-01; 2026-07-02 | `git log -S "A12 How To Build"`; extended during 253/Popometer/dashboard synthesis |
| A11 | 2026-07-01 | `git log -S "A11 Preserve Good"` |
| A10 | 2026-07-01 | `git log -S "A10 10D Gives"` |
| A9 | 2026-07-01 | `git log -S "A9 Earth Is Learnable"` |
| A8 | 2026-07-01 | `git log -S "A8 Hard Worlds"` |
| A7 | 2026-07-01 | `git log -S "A7 Good HPs"` |
| A6 | 2026-07-01 | `git log -S "A6 Observation Mode"` |
| A5 | 2026-07-01 | `git log -S "A5 Visualize Early"` |
| A4 | 2026-07-01 | `git log -S "A4 Let Optuna"` |
| A3 | 2026-07-01 | `git log -S "A3 Gamma And Tau"` |
| A2 | 2026-07-01 | `git log -S "A2 Back Up"` |
| A1 | 2026-06-29 | `git log -S "A1 Code Complexity"` |
| H1 | 2026-07-01 | `git log -S "H1 Earth Is Learnable"` |
| H2 | 2026-07-01 | `git log -S "H2 Hard Worlds"` |
| H3 | 2026-07-01 | `git log -S "H3 Sampling Should"` |
| H4 | 2026-07-01 | `git log -S "H4 Observation Mode"` |
| H5 | 2026-07-01 | `git log -S "H5 Good HPs"` |
| Q2 | 2026-07-01 | `git log -S "Q2 Training Nutzen"` |
| Q1 | 2026-07-01 | `git log -S "Q1 SolarSystemLander Difficulty"` |
| Pilot Preservation Ideas | 2026-06-30; 2026-07-02 | `git log -S "Pilot Preservation Ideas"`; early talent signal added on 2026-07-02 |
| Video vom Training | 2026-06-30 | `git log -S "Video vom Training"` |
| Dashboard liest aus DB | 2026-06-30 | `git log -S "Dashboard liest aus DB"` |
| Begriffe aus dem Apollo-Programm | 2026-06-29 | `git log -S "Begriffe aus dem Apollo"` |
| E1 | 2026-07-02 | created directly in `ouch.md` |
