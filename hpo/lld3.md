# SolarSystemLander HPO LLD

Dieses LLD beschreibt die Umsetzung von `hpo/design3.md` für die Studienreihen 2A und 2B.

## Zielstruktur

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

## Gemeinsame Objective

`hpo.lunar_lander.objective` wird nach `hpo.objective` verschoben und für LunarLander und SolarSystemLander gemeinsam verwendet.

`create_objective(...)` führt den gemeinsamen Trial-Ablauf aus:

1. Trainingskonfiguration und Replay-Kapazität aus dem Search Space bestimmen.
2. Trainings-Environment über die Environment-Fabrik erzeugen.
3. Ein Q-Netz mit dem bestehenden `VectorTrainer` trainieren.
4. Die benannten Eval-Environments greedy bewerten.
5. Den Mittelwert `gym_score` und den Quality-Effort Score berechnen.
6. Messwerte und `gym_scores` als Trial-Attribute speichern.

Die Objective behandelt die Namen in `gym_scores` generisch. LunarLander liefert einen Eintrag, SolarSystemLander vier Einträge für Mond, Mars, Erde und Venus.

## Environment-Fabriken

Die aufgabenspezifischen Unterschiede werden durch je eine konfigurierte Environment-Fabrik gekapselt:

```python
class EnvironmentFactory(Protocol):
    def make_training_vector_env(self, *, num_envs) -> VectorEnv: ...
    def evaluation_env_factories(self) -> dict[str, Callable[[], Any]]: ...
```

`hpo.lunar_lander.environment` erzeugt `LunarLander-v3` für Training und Evaluation.

`hpo.solar_system_lander.environment` erzeugt die Umgebungen für Mond, Mars, Erde und Venus. Es kapselt die Gravitation, das reproduzierbare episodische Wetter sowie den beim Erzeugen der Fabrik gewählten Observation-Modus 8D oder 11D.

Das SSL-Trainings-Environment verwendet 16 Vector-Slots mit vier Slots je Himmelskörper. Alle Erfahrungen gelangen in das gemeinsame Replay Memory des `VectorTrainer`.

## Gemeinsame Greedy-Evaluation

`evaluate_greedy_q_net(...)` wird aus der bisherigen LunarLander-Objective nach `hpo.evaluation.greedy` verschoben. Die Funktion erhält eine `make_env`-Funktion und kann dadurch alle Eval-Environments unabhängig von Observation-Größe und Szenario ausführen.

Die gemeinsame Objective ruft sie für jedes benannte Eval-Environment auf und speichert die Ergebnisse als `gym_scores`.

## Gemeinsame Study-Orchestrierung

`hpo.lunar_lander.study` wird nach `hpo.study` verschoben. `run_study(...)` und `select_robust_best(...)` verwenden die konfigurierte gemeinsame Objective und einen expliziten SQLite-Pfad.

Damit verwenden LunarLander und SolarSystemLander dieselbe Optuna-Steuerung, Baseline-Berechnung und robuste Auswahl.

## Notebook und Studienreihen

Das neue `HPO_SolarSystemLander.ipynb` steuert beide Studienreihen. Eine zentrale Einstellung wählt den Observation-Modus:

```python
OBSERVATION_MODE = "8d"   # Series 2A
# OBSERVATION_MODE = "11d"  # Series 2B
```

Beide Reihen verwenden denselben Suchraum, dieselben Seeds und die Gewinner-Hyperparameter aus Series 1 als Ausgangskonfiguration. Sie laufen auf getrennten L4-Runtimes und speichern in getrennte Datenbanken:

```text
solar_system_lander_8d.db
solar_system_lander_11d.db
```

Das bestehende LunarLander-Notebook wird an die gemeinsamen Objective- und Study-Schnittstellen angepasst.

## Bezug zum Sequenzdiagramm

Der Ablauf aus `hpo/sequence_diagram.md` gilt weiterhin:

```text
Notebook → Study → Objective → VectorTrainer → Evaluation → Objective-Score
```

Geändert werden die beteiligten Implementierungen: `study.py` und `objective.py` liegen im allgemeinen `hpo`-Package, die Objective erhält eine Environment-Fabrik, und die Greedy-Evaluation läuft über die benannten Eval-Environments dieser Fabrik. Der dargestellte Kontrollfluss und die Trial-Schleife bleiben gleich.
