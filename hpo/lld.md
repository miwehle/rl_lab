# Lunar Lander HPO LLD

Dieses LLD beschreibt die Umsetzung des HLD aus `hpo/design.md`.

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
- Trial-Score berechnen:

```python
best_window = best_window_mean(result.episode_returns, score_window)
best_window_score = best_window.mean
final_returns = result.episode_returns[-score_window:]
final_window_score = sum(final_returns) / len(final_returns)
objective_score = (best_window_score + final_window_score) / 2
```

- `best_window_score`, `final_window_score`, `objective_score` und
  `best_window_*` als Trial-Attribute speichern.
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
    num_episodes,
    score_window,
    output_dir,
    study_dir,
    device,
    num_envs=16,
    seed=42,
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
    num_episodes,
    score_window,
    device,
    num_envs=16,
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

### `hpo.evaluation.reporting`

Neues Diagramm:

```python
plot_lander_progress(study)
```

Inhalt:
- x-Achse: kumulierte Trainingszeit auf L4.
- y-Achse: Greedy-Eval-Score (`epsilon=0`).
- horizontale Marken bei `200` und `250`.

Die Objective speichert dafür pro Trial:
- `wall_time_seconds` (nur Training, ohne Greedy Eval)
- `eval_score`

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
- `LH`-x: mittlere Trainingszeit pro Trial der Studie; Robustheitsprüfungen zählen nicht mit.
- `LH`-y: `eval_score` des besten/robusten Studienergebnisses.
- `OH`: Optuna History der aktuell laufenden Studie, ein Punkt pro Trial.

Die Ausgabe wird nach jedem Trial aktualisiert. Frühere Live-Ausgaben werden
gelöscht; am Ende bleibt der letzte Stand sichtbar.

## Nicht Ziel

- Kein Pruning.
- Keine generische Objective für mehrere Trainer.
- Keine SearchSpace-Abstraktion außerhalb des Notebooks, solange die Suchräume
  noch in Bewegung sind.

## Umsetzungsphasen

### Phase 1: Package-Kern

- `create_objective(...)` direkt auf `VectorTrainer` umbauen.
- Scoring gemäß HLD umsetzen.
- Greedy Eval mit `eval_episodes=3`.
- Trial-Attribute für Score, Eval-Score und Trainingszeit speichern.
- Objective-Tests anpassen.

Das HPO-Notebook darf in dieser Phase temporär nicht laufen.

### Phase 2: Orchestrierung und Reporting

- `run_study(...)`, `select_robust_best(...)`, `neighbors(...)` implementieren.
- `plot_lander_progress(study)` implementieren.
- Tests für Study-Helfer und Reporting ergänzen.

### Phase 3: Notebook

- HPO-Notebook auf `VectorTrainer`-only umbauen.
- `SearchSpace1` bis `SearchSpace4` eintragen.
- Studienfolge mit `run_study(...)` starten.
- Nach jeder Studie `plot_lander_progress(study)` anzeigen.

### Phase 4: Live-Plotting

- `SearchSpace0` und `study0` vor S1 ergänzen.
- Notebook-Ausgabe während HP4 auf genau zwei Diagramme umstellen.
- `LH`: Study-History (`S0 -> S1 -> S2 -> S3 -> S4`).
- `OH`: Optuna History der aktuellen Study.
- Beide Diagramme nach jedem Trial aktualisieren.
