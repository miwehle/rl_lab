# HPO User Manual

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

## Connect Storage

### Local Runtime

### Google Drive

### Study Database

## Define The Study Setup

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
