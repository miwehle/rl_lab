# SolarSystemLander HPO – Low-Level Design

Dieses LLD beschreibt die Implementierung von `hpo/design3.md`. Series 2A und 2B verwenden dieselbe Implementierung; nur der Observation-Modus und die SQLite-Datei unterscheiden sich.

## Leitlinien

- Ein gemeinsames Q-Netz lernt aus allen fünf Welten.
- `VectorTrainer`, Replay Memory und DQN bleiben unverändert.
- Aufgabenunterschiede werden durch Environment-Fabriken und Wrapper gekapselt.
- Gemeinsamer HPO-Ablauf wird nicht für LunarLander und SolarSystemLander dupliziert.
- Series 2A und 2B verwenden identische Suchräume, Seeds und Auswertung.

## Zielstruktur

```text
hpo/
├── notebooks/
│   ├── HPO_LunarLander.ipynb
│   └── HPO_SolarSystemLander.ipynb       +
└── src/hpo/
    ├── objective.py                      +  gemeinsame Objective
    ├── study.py                          +  gemeinsame Study-Orchestrierung
    ├── evaluation/
    │   ├── greedy.py                     +  gemeinsame Greedy-Evaluation
    │   └── scoring.py                    =  Quality-Effort Score
    ├── lunar_lander/
    │   ├── environment.py                +
    │   └── logging.py                    =
    └── solar_system_lander/
        ├── __init__.py                   +
        └── environment.py                +
```

Die bisherige LunarLander-Objective und Study werden in die gemeinsamen Module verschoben. Imports im bestehenden Notebook und in den Tests werden angepasst; es bleiben keine Kompatibilitätsmodule zurück.

## Environment-Schnittstelle

Die gemeinsame Objective kennt keine Planetenparameter. Sie erhält eine konfigurierte Task-Fabrik:

```python
class EnvironmentFactory(Protocol):
    def make_training_env(self, num_envs: int): ...
    def evaluation_envs(self) -> dict[str, Callable[[], Any]]: ...
```

`lunar_lander.environment` liefert die bisherige einzelne `LunarLander-v3`-Umgebung. `solar_system_lander.environment` liefert die fünf Welten und kapselt Observation-Modus, Wetterverteilungen und Metadaten.

## SolarSystemLander-Environments

Eine `WorldConfig` als `@dataclass(frozen=True)` beschreibt Name, Gravitation sowie Intervalle für `wind_power` und `turbulence_power`. Nach dem Erzeugen können ihre Felder nicht versehentlich verändert werden. Die fünf Konfigurationen entsprechen der Tabelle in `design3.md`.

Das Training verwendet eine `SyncVectorEnv` mit einer durch fünf teilbaren Slot-Anzahl, zunächst 20 Slots: vier Slots pro Welt. Dadurch gelangen pro Vector-Step gleich viele Transitions jeder Welt in das gemeinsame Replay Memory. `num_episodes` beendet das Training nach der angegebenen Gesamtzahl abgeschlossener Episoden, nicht je Welt.

### Episodisches Wetter

Ein kleiner Wrapper zieht bei jedem `reset()` reproduzierbar `wind_power` und `turbulence_power` aus den Intervallen der Welt und setzt die Werte vor dem Reset der Gymnasium-Umgebung. Mond und Merkur verwenden `enable_wind=False`.

Der Wrapper hat einen eigenen, über den Environment-Seed initialisierten Zufallsgenerator. Autoresets ohne neuen Seed setzen dessen deterministische Zahlenfolge fort. Gleiche Trial- und Eval-Seeds erzeugen damit dieselben Wetterfolgen.

### Observation-Modi

- `8d`: Die originale Gymnasium-Observation wird unverändert weitergegeben.
- `11d`: Ein `ObservationWrapper` hängt die für die aktuelle Episode gezogenen Werte `gravity`, `wind_power` und `turbulence_power` an.

Die drei zusätzlichen Werte werden auf vergleichbare Größenordnungen normiert:

```text
gravity / 12
wind_power / 20
turbulence_power / 2
```

Die 11D-Observation enthält die maximalen Wetterstärken der Episode, nicht die momentane Windkraft. Das DQN erkennt die Eingabegröße bereits aus dem Observation Space; `dqn.model.DQN` benötigt keine Änderung.

## Gemeinsame Objective

`hpo.objective.create_objective(...)` führt pro Trial aus:

1. Trainingskonfiguration einschließlich `num_episodes` aus dem Search Space bestimmen.
2. Replay-Kapazität bestimmen und das gemischte Vector-Environment erzeugen.
3. Ein gemeinsames Q-Netz mit dem bestehenden `VectorTrainer` trainieren.
4. Das finale Q-Netz auf jeder Welt greedy evaluieren.
5. `g` als gleichgewichteten Mittelwert der fünf Welt-Scores berechnen.
6. Trainingsaufwand `t` und Quality-Effort Score `o` wie bisher berechnen.
7. Rohwerte entsprechend der bestehenden Quality-Effort-Persistenz speichern.

Die in `design2.md` definierte und bereits implementierte Persistenz bleibt erhalten:

```text
trial.value                # o
gym_score                  # g
env_steps
optimizer_updates
processed_samples
training_effort
trial_seed
wall_time_seconds
training_curve
```

Für den SolarSystemLander kommt als Trial-Attribut hinzu:

```text
gym_scores                 # {moon, mercury, mars, earth, venus}
```

Ein redundantes `objective_score`-Attribut wird weiterhin nicht gespeichert. Die robuste Auswahl vergleicht wie bisher `trial.value`, also den Quality-Effort Score `o`.

Die vorhandene `evaluate_greedy_q_net(...)` wandert nach `hpo.evaluation.greedy` und erhält eine parameterlose `make_env`-Funktion statt einer festen Environment-ID.

Die Objective evaluiert jede Welt mit dem in `design3.md` festgelegten Episoden- und Seed-Satz.

## Search Space und `num_episodes`

`num_episodes` darf nicht länger ausschließlich in `TrialConfig` stehen, weil es in Series 2 ein HP ist. Das `SearchSpace` erzeugt die vollständige `VectorTrainingConfig` einschließlich `num_episodes`.

`TrialConfig` enthält danach nur noch Ausführungsparameter:

```text
num_envs
seed
device
```

Die Search-Space-Klassen bleiben wie bisher notebooknah und bilden exakt die Tabelle aus `design3.md` ab. `_FixedParamTrial` kann `num_episodes` bei der robusten Nachprüfung ohne Sonderfall wiederverwenden.

## Study-Orchestrierung und Speicherung

`hpo.study.run_study(...)` erhält die Environment-Fabrik, einen expliziten SQLite-Pfad und optionale aufgabenspezifische Study-Attribute. Alle Studies einer Series werden in derselben Datei gespeichert:

```text
solar_system_lander_8d.db
solar_system_lander_11d.db
```

Die bestehende Quality-Effort-Persistenz bleibt unverändert: Study-Attribute speichern Scoring-Konfiguration und Eval-Seeds; S0 speichert zusätzlich `baseline_env_steps` und `baseline_processed_samples`. Für den SolarSystemLander kommen Observation-Modus und World-Konfigurationen hinzu. Beim Fortsetzen werden alle Attribute wie bisher auf Übereinstimmung geprüft.

S0 wird in jeder Series mit den Gewinner-HPs aus Study Series 1 neu ausgeführt. Seine `env_steps` und `processed_samples` bilden die jeweilige Effort-Baseline für S1 bis S4.

## Notebook

`HPO_SolarSystemLander.ipynb` enthält eine zentrale Auswahl:

```python
OBSERVATION_MODE = "8d"   # Series 2A
# OBSERVATION_MODE = "11d"  # Series 2B
```

Daraus werden Environment-Fabrik und Datenbankpfad abgeleitet. Beide Colab-Runtimes führen dasselbe Notebook mit unterschiedlichem Modus aus. Studienfolge, Seeds, Search Spaces und Analyse bleiben identisch.

Das vorhandene `HPO_LunarLander.ipynb` wird nur auf die gemeinsamen Imports und Schnittstellen angepasst. Inhalt, Ausführung und bisherige Ergebnisse bleiben funktional kompatibel.

## Tests

Gezielte Tests sichern:

- reproduzierbares episodisches Wetter,
- 8D- und 11D-Observation Space,
- gleichmäßige Verteilung der Vector-Slots auf fünf Welten,
- Mittelung und Speicherung der fünf Welt-Scores,
- `num_episodes` aus dem Search Space,
- gemeinsame SQLite-Datei mit getrennten Study-Namen,
- unveränderte LunarLander-HPO über die gemeinsame Objective,
- weiterhin vollständig ausführbares `HPO_LunarLander.ipynb`.

Nicht Teil dieser Umsetzung sind PER, Teacher-Student-Training, neue DQN-Architekturen oder Reward-Anpassungen.
