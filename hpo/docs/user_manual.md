# HPO User Manual

---
**Vorab**: Status dieses Manuals: 10% fertige Blockhütte in der chaotischen Entstehung.

> [!quote] Codex:
> Später können wir die anderen Bausteine drumherum sortieren: Flow, Setup, Optimize, Analyze, Appendix.

Das schöne Dashboard hat Codex nicht genannt (der wertschätzt das stets geringer als ich).
Naja, man kann es als wichtiges Tool innerhalb von Optimize sehen (vielleicht meinte er es so).

---

Purpose: a short handout for using the `hpo` package from notebooks or small Python modules.


Spiel starten:

`.\dqn\.venv\Scripts\python.exe -m hpo.games.solar_system_lander`

↑        Haupttriebwerk
← / →    Seitendüsen
Space    Boost, also feuern in jedem Step statt gepulst
R        Restart gleiche Welt/Seed
N        nächster Seed
1..5     Moon, Mercury, Mars, Earth, Venus
Esc      Ende


## What HPO Does

## Typical Notebook Flow

## Define The Study Setup

### Convention over Configuration

You normally do not configure HPO infrastructure in notebooks. HPO uses convention over configuration for local paths, Drive paths, database/log files, checkpoints, videos, and artifact layout.

Define the study choices here: study name, environment, worlds, baseline/incumbent, search space, objective settings, and dashboard. If you need to change the infrastructure convention, override `InfraCfg`; see [Appendix: Infrastructure Overrides](#appendix-infrastructure-overrides).

### Environment Factory

### Objective Config

### Baseline

Best practice: keep the final parameter set complete: baseline/incumbent plus `suggest_*` values must contain every HP needed to build and train the agent.

Use `suggest_*` only for HPs that the current study should actually vary. HPs optimized by Optuna may be omitted from the baseline because `trial.params` will provide them. If a HP should stay fixed in this study, read it from the baseline/incumbent instead of passing it through `suggest_*` with a single constant value.

This keeps Optuna's `trial.params` meaningful: it roughly means "Optuna optimized this HP in this study." The dashboard relies on that simple interpretation when highlighting current HPs.

### Dashboard

## Run A Study

### Start A New Study

### Continue A Study Series

### Change The Search Space

## Inspect Results

### Current Incumbent

### Best Checkpoints

### Study Database

## Robustness Evaluation

### HP Robustness

### Checkpoint Robustness

## Colab Reconnects

## Minimal Examples

## Appendix: Infrastructure Overrides

HPO's notebook-facing APIs use default infrastructure conventions. Treat `InfraCfg` as an escape hatch, not as part of the normal study setup.

`StudyRunner` uses its default study infrastructure config unless you pass another one:

```python
runner = StudyRunner(...)
runner = StudyRunner(..., cfg=custom_study_infra_cfg)
```

Video recording follows the same idea:

```python
record_video(...)
record_video(..., cfg=custom_video_infra_cfg)
```

Override `cfg` only for infrastructure concerns such as storage roots, artifact names, directory layout, or runtime-specific paths. Keep domain choices outside `InfraCfg`: study names, worlds, seeds, models, training parameters, objective settings, and search spaces stay explicit in the notebook.
