# SolarSystemLander HPO LLD

Dieses LLD beschreibt die Umsetzung von `hpo/design3.md`. Der grundsätzliche Trial-Ablauf bleibt wie in `hpo/sequence_diagram.md`; neu sind das SolarSystem-Environment und die Bewertung über vier Himmelskörper.

## Ziel

Die Studienreihen (study series) 2A und 2B trainieren mit dem bestehenden `VectorTrainer` jeweils ein gemeinsames Q-Netz für Mond, Mars, Erde und Venus. Beide Reihen verwenden denselben Suchraum und dieselben Seeds; sie unterscheiden sich nur durch die Observation mit 8D beziehungsweise 11D.

## Python-Package

Zielstruktur der betroffenen Dateien:

```text
hpo/
├── notebooks/
│   ├── HPO_LunarLander.ipynb       ~
│   └── HPO_SolarSystemLander.ipynb +
└── src/hpo/
    ├── objective.py                →
    ├── study.py                    →
    ├── evaluation/
    │   ├── greedy.py               +
    │   ├── scoring.py              =
    │   └── reporting.py            =
    ├── lunar_lander/
    │   ├── environment.py          +
    │   └── logging.py              =
    └── solar_system_lander/
        ├── __init__.py             +
        └── environment.py          +
```

`+` neu · `~` geändert · `=` unverändert · `→` verschoben

`hpo.objective` enthält den gemeinsamen Trial-Ablauf, `hpo.study` die gemeinsame Optuna-Orchestrierung. Die aufgabenspezifische Logik bleibt in den beiden Environment-Fabriken.

`hpo.lunar_lander.objective` wird nach `hpo.objective` verschoben und für beide Aufgaben verallgemeinert. Die neue LunarLander-Fabrik kapselt lediglich die Erzeugung des unveränderten Gymnasium-Environments. Die SolarSystemLander-Fabrik kapselt zusätzlich Himmelskörper, episodisches Wetter, 8D/11D-Observations und die gemischte Vector-Umgebung.

Das LunarLander-Notebook wird an die verallgemeinerte Schnittstelle von `hpo.study` angepasst.

### Environment-Fabriken

Die gemeinsame Objective kennt keine konkreten Gymnasium-Environments. Sie verwendet eine kleine aufgabenspezifische Fabrik mit zwei Verantwortungen:

```python
class EnvironmentFactory(Protocol):
    def make_training_vector_env(self, *, num_envs, seed) -> VectorEnv: ...
    def evaluation_env_factories(self) -> dict[str, Callable[[], Any]]: ...
```

`make_training_vector_env(...)` liefert ein Gymnasium `VectorEnv` mit `num_envs` Teil-Environments für das Training. `evaluation_env_factories()` liefert benannte Fabriken für die Greedy-Evaluation; die Zahl der Evaluationsepisoden bestimmt die Scoring-Konfiguration.

LunarLander liefert ein Eval-Environment:

```text
lunar_lander
```

SolarSystemLander liefert vier:

```text
moon
mars
earth
venus
```

Weitere Hooks oder eine Objective-Klassenhierarchie werden nicht eingeführt.

### `hpo.lunar_lander.environment`

Die LunarLander-Fabrik erzeugt für Training und Evaluation direkt `LunarLander-v3`. Sie enthält keine eigene Environment-Logik; das kleine Modul stellt lediglich dieselbe Fabrikschnittstelle wie SolarSystemLander bereit.

### `hpo.solar_system_lander.environment`

Das neue Modul kapselt die aufgabenspezifischen Umgebungen.

- Die vier Himmelskörper und ihre Wetterintervalle werden als feste Szenariodefinitionen hinterlegt.
- Ein Environment bleibt einem Himmelskörper zugeordnet und zieht bei jedem `reset()` Wind und Turbulenz reproduzierbar aus seinem Seed-RNG.
- Mond verwendet Wind und Turbulenz `0`; bei den anderen Körpern aktiviert das Environment Gymnasiums Windmodell.
- 8D liefert die unveränderte LunarLander-Observation.
- 11D hängt normalisierte Werte für `gravity`, `wind_power` und `turbulence_power` an. Verwendet werden `gravity / 12`, `wind_power / 20` und `turbulence_power / 2`.
- Das Vector-Environment verteilt seine Slots gleichmäßig auf die vier Körper. Bei `num_envs=16` sind dies vier Slots je Körper.

Series 2A und 2B verwenden `num_envs=16` mit vier Slots je Himmelskörper. Eine dynamische Körperwahl im Trainer ist nicht erforderlich.

### `hpo.objective`

Die gemeinsame `create_objective(...)` erhält eine Environment-Fabrik und führt für LunarLander und SolarSystemLander denselben Trial-Ablauf aus:

1. Trainingskonfiguration und Replay-Kapazität aus dem Search Space bestimmen.
2. Trainings-Environment über die Fabrik erzeugen.
3. Ein gemeinsames Q-Netz mit dem bestehenden `VectorTrainer` trainieren.
4. Alle benannten Eval-Environments der Fabrik mit festen Seeds greedy evaluieren.
5. Den gleichgewichteten Mittelwert der Einzelscores bilden.
6. Den bestehenden Quality-Effort Score berechnen und zurückgeben.

Die Objective speichert die benannten Einzelscores generisch als Map und ihren Mittelwert separat:

```text
gym_scores = {
    "moon": 240.0,
    "mars": 215.0,
    "earth": 195.0,
    "venus": 170.0,
}
gym_score = 205.0
```

Für LunarLander enthält `gym_scores` entsprechend nur den Eintrag `lunar_lander`. Die Objective kennt die fachliche Bedeutung der Namen nicht. `gym_score` ist das arithmetische Mittel aller Werte aus `gym_scores` und geht in den Quality-Effort Score ein.

### `hpo.evaluation.greedy`

Die bestehende Funktion `evaluate_greedy_q_net(...)` wird aus der LunarLander-Objective in dieses gemeinsame Modul verschoben. Sie erhält eine einzelne `make_env`-Funktion statt einer festen `env_id` und bleibt unabhängig von Observation-Größe, Himmelskörpern und Wetter.

LunarLander ruft den Greedy-Rollout einmal auf, SolarSystemLander viermal. Die Zusammenführung der Scores übernimmt die gemeinsame Objective. Die vier SolarSystemLander-Einzelscores werden zunächst im Notebook dargestellt; dafür wird keine zusätzliche Reporting-Abstraktion eingeführt.

### Study-Orchestrierung

Die vorhandene Study-Orchestrierung wird von `hpo.lunar_lander.study` nach `hpo.study` verschoben und nur so weit verallgemeinert, dass `run_study(...)` und `select_robust_best(...)` eine konfigurierte Objective verwenden können und der SQLite-Pfad unabhängig vom Study-Namen angegeben werden kann.

```python
run_study(
    *,
    study_name,
    storage_path,
    objective_factory,
    ...
)
```

Die übrige Optuna-Steuerung, Baseline-Berechnung und robuste Auswahl bleiben unverändert. Es entsteht weder eine zweite Kopie der Study-Orchestrierung noch eine Klassenhierarchie. Bestehende Imports und Tests werden auf `hpo.study` umgestellt.

## Training und Ablauf

Der Ablauf aus `hpo/sequence_diagram.md` bleibt gültig. Die gemeinsame Objective lässt das Trainings-Environment von der gewählten Fabrik erzeugen und wertet nach dem Training deren benannte Eval-Environments aus. `vector_training.py` bleibt unverändert.

Das bestehende Vector-Training erfüllt bereits die Anforderungen:

- ein gemeinsames Q-Netz,
- ein gemeinsamer Replay-Speicher,
- gemischte Erfahrungen aus allen Vector-Slots,
- Unterstützung für flache 8D- und 11D-Observations.

Ein `MultiPlanetTrainer`, getrennte Replay-Speicher oder sequenzielles Training pro Körper werden nicht eingeführt.

## Notebook

Ein neues Notebook `hpo/notebooks/HPO_SolarSystemLander.ipynb` steuert beide Reihen. Das bestehende LunarLander-Notebook bleibt die ausführbare Dokumentation von Series 1 und wird nur an die gemeinsamen Objective- und Study-Schnittstellen angepasst.

Eine zentrale Einstellung wählt die Reihe:

```python
OBSERVATION_MODE = "8d"  # L4 #1
# OBSERVATION_MODE = "11d"  # L4 #2
```

Daraus folgt der Datenbankpfad:

```text
8d  -> solar_system_lander_8d.db
11d -> solar_system_lander_11d.db
```

Das Notebook enthält:

- die Gewinner-Hyperparameter aus Series 1 als Ausgangskonfiguration,
- die Baseline für die neue Vier-Körper-Aufgabe,
- den gemeinsamen Search Space für 2A und 2B,
- Start und Fortsetzung der Optuna-Studies,
- eine Ergebnistabelle mit Gesamtscore und Score je Körper.

Beide L4-Runtimes führen dasselbe Notebook mit unterschiedlichem `OBSERVATION_MODE` aus. Zwei nahezu identische Notebooks werden vermieden.

## Persistenz

Jede Reihe verwendet genau eine eigene SQLite-Datenbank. Baseline und HPO sind getrennte Optuna-Studies innerhalb dieser Datenbank. Dadurch teilen sie ihre Datenbasis, ohne dass 2A und 2B konkurrierend auf dieselbe Datei schreiben.

Die Trial-Attribute speichern `gym_scores` als benannte Map und `gym_score` als gemeinsamen Mittelwert. Die Study-Attribute speichern zusätzlich den Observation-Modus und die verwendeten Himmelskörper. Die bisherigen Scoring-, Baseline- und Robustheitsattribute bleiben erhalten.

## Tests

- Die gemeinsame Objective funktioniert mit der LunarLander- und der SolarSystemLander-Fabrik.
- Der gemeinsame Greedy-Rollout verwendet die übergebene Eval-Environment-Funktion und feste Seeds.
- Wetterwerte liegen in den definierten Intervallen und sind mit gleichem Seed reproduzierbar.
- 8D- und 11D-Observations haben die erwartete Form und die Zusatzwerte sind normalisiert.
- Das Vector-Environment verteilt seine Slots gleichmäßig auf vier Körper.
- Die Objective speichert beliebig benannte Einzelscores als `gym_scores` und bildet daraus den korrekten `gym_score`.
- Series 2A und 2B verwenden unterschiedliche SQLite-Dateien.

## Vor dem Studienstart festzulegen

- Die endgültigen Gewinner-Hyperparameter aus Series 1 werden aus der vorhandenen Optuna-Datenbank übernommen und im Notebook dokumentiert.
- Die Zahl der Eval-Episoden wird als Anzahl pro Körper eindeutig festgelegt. Empfohlen sind 20 feste Seeds pro Körper.
