# Lunar Lander HPO LLD

Dieses LLD beschreibt die Umsetzung des HLD aus `hpo/designs/design.md`.

## Ziel

Das HPO-Notebook startet die Studienfolge für LunarLander mit dem
`VectorTrainer`. Andere Trainer werden im HPO-Pfad nicht mehr unterstützt.

## Python-Package

### `hpo.lunar_lander.objective`

`create_objective(...)` wird auf den `VectorTrainer` ausgerichtet.

Aufgaben:
- Vector-Environment erzeugen, z. B. `SyncVectorEnv` mit `num_envs=16`.
- `search_space.training_config(trial, num_episodes)` liefert eine
  `VectorTrainingConfig`.
- `search_space.replay_memory_capacity(trial)` liefert die Replay-Kapazität.
- `VectorTrainer(...).train(training_config)` ausführen.
- Greedy Gym-Score mit 20 festen Eval-Seeds berechnen.
- Hardwareunabhängigen Trainingsaufwand relativ zu S0 berechnen:

```python
processed_samples = result.optimizer_updates * training_config.batch_size
effort = training_effort(
    env_steps=result.env_steps,
    processed_samples=processed_samples,
    baseline_env_steps=baseline_env_steps,
    baseline_processed_samples=baseline_processed_samples,
    alpha=alpha,
)
quality = (gym_score - quality_min) / (quality_target - quality_min)
objective_score = quality_weight * quality - (1 - quality_weight) * (effort - 1)
```

- Rohwerte und Aufwand als Trial-Attribute speichern.
- `objective_score` zurückgeben.

Kein Pruning im VectorTrainer-HPO-Pfad.

### `hpo.lunar_lander.study`

Neue Funktion:

```python
run_study(
    *,
    study_name,
    search_space,
    n_trials,
    study_dir,
    trial_cfg,
    scoring_cfg,
)
```

Aufgaben:
- `objective = create_objective(...)` erzeugen.
- Optuna-Study mit SQLite-Storage erzeugen oder laden.
- Bis `n_trials` abgeschlossene Trials erreicht sind:
  - `study.optimize(objective, n_trials=1)`
  - Fortschritt anzeigen.
- `study` zurückgeben.

Weitere Funktionen:

```python
select_robust_best(
    *,
    study,
    search_space_factory,
    trial_cfg,
    scoring_cfg,
    base_seed=42,
    top_n=3,
    extra_seeds=(1001, 1002),
)
neighbors(value, choices)
```

`select_robust_best(...)` implementiert die robuste Top-3/Seed-Prüfung aus dem
HLD: Top-Kandidaten erneut mit Zusatz-Seeds trainieren, mitteln, beste
HP-Kombination zurückgeben. `neighbors(...)` liefert einen Wert plus direkte
Nachbarn aus einer geordneten Menge.

`run_study(...)` kapselt Optuna-Details, damit das Notebook nur Studien steuert.

### `hpo.evaluation.dashboard`

Das Notebook-Dashboard zeigt den Fortschritt der Study Series, die aktuellen besten Hyperparameter, die laufende Studie und deren robuste Kandidatenprüfung.

## Notebook

Das Notebook bleibt die Steuerzentrale.

### Setup-Zellen

Importe anpassen:
- `VectorTrainingConfig` statt `TrainingConfig`/`TuningConfig`.
- `run_study`, `select_robust_best`, `neighbors` aus `hpo.lunar_lander.study`.

HPO-Parameter zentral setzen:

```python
NUM_EPISODES = 600
SCORE_WINDOW = 100
NUM_ENVS = 16
SEED = 42
```

### Search-Spaces

Je Studie eine eigene Klasse im Notebook:

```python
SearchSpace0
SearchSpace1
SearchSpace2
SearchSpace3
SearchSpace4
```

Die Klassen bilden die Matrix aus `design.md` ab.
`SearchSpace0` setzt feste Baseline-HP und nutzt keine `trial.suggest_*`-Aufrufe.

Abhängigkeiten:
- `SearchSpace2(best_s1)`
- `SearchSpace3(best_s1, best_s2)`
- `SearchSpace4(best_s1, best_s2, best_s3)`

### Studienfolge

Das Notebook ruft nacheinander auf:

```python
study0 = run_study("s0_baseline", SearchSpace0(), 1, ...)

study1 = run_study("s1_update_economy", SearchSpace1(), 40, ...)
best_s1 = select_robust_best(study1)

study2 = run_study("s2_exploration", SearchSpace2(best_s1), 40, ...)
best_s2 = select_robust_best(study2)

study3 = run_study("s3_replay_capacity", SearchSpace3(best_s1, best_s2), 10, ...)
best_s3 = select_robust_best(study3)

study4 = run_study("s4_joint_finetune", SearchSpace4(best_s1, best_s2, best_s3), 30, ...)
```

`select_robust_best(...)` führt die Robustheitsprüfung gemäß HLD aus und gibt
die beste HP-Kombination zurück.

### Studie 0 und Diagramme

- `SearchSpace0`: feste Baseline-HP, keine `trial.suggest_*`-Aufrufe, `n_trials=1`.
- `LH`: Study-History mit einem Punkt pro Studie (`S0` bis `S4`).
- `LH`-x: robuster Trainingsaufwand des gewählten Studienergebnisses.
- `LH`-y: robuster Greedy-Gym-Score desselben Ergebnisses.
- `OH`: Optuna History der aktuell laufenden Studie, ein Punkt pro Trial.

Die Ausgabe wird nach jedem Trial aktualisiert. Frühere Live-Ausgaben werden
gelöscht; am Ende bleibt der letzte Stand sichtbar.

## Persistenz

Die Optuna-SQLite-DB ist die primäre Datenquelle.
In Colab liegt die aktive SQLite-DB während des Trainings lokal im
Colab-Dateisystem. Nach jeder Studie wird sie nach Google Drive kopiert. Beim
Neustart wird eine vorhandene Drive-DB zurück nach lokal kopiert.

### Custom Attributes

Trial-User-Attrs:
- `gym_score`: Greedy-Gym-Score nach dem Training.
- `env_steps`, `optimizer_updates`, `processed_samples`: Aufwandsrohwerte.
- `training_effort`: relativ zu S0 normalisierter Trainingsaufwand.
- `wall_time_seconds`: reine Trainingszeit, ohne Greedy Eval.
- `training_curve`: Returns und Epsilons je Episode.

Study-User-Attrs:
- Scoring-Konfiguration und feste Eval-Seeds.
- `baseline_env_steps`, `baseline_processed_samples`.
- `robust_best_params`: robust beste HP-Kombination der Studie.
- `robust_best_objective_score`: robuster Objective-Score dieser Kombination.
- `robust_best_gym_score`: robuster Greedy-Gym-Score dieser Kombination.
- `robust_best_training_effort`: robuster Aufwand dieser Kombination.

Nutzung:
- `trial.value` steuert die Optuna-Optimierung.
- robuste Study-Attrs speisen `LH`.
- `training_curve` rekonstruiert später das Trainingsdiagramm aus dem
  DQN-Notebook: Returns pro Episode, geglättete Return-Kurve und Epsilon-Kurve.
  Die geglättete Kurve wird nicht gespeichert, sondern aus den Rohdaten
  berechnet.

Buffering:
- Es gibt keine DB-Writes pro Episode. Die Zeitreihen werden im RAM gesammelt und einmal am Ende des Trials geschrieben.

## Nicht Ziel

- Kein Pruning.
- Keine generische Objective für mehrere Trainer.
- Keine SearchSpace-Abstraktion außerhalb des Notebooks, solange die Suchräume
  noch in Bewegung sind.
