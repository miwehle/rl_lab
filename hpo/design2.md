# LunarLander HPO – Scoring und Training Economy

Dieser Entwurf entwickelt `design.md` weiter. Das Ziel aus
*The Affordable LunarLander Ferrari – A Quest* bleibt:

> Hoher Greedy-Eval-Score bei möglichst geringem Trainingsaufwand.

## 1. Quality-Effort Score

Der bisherige Optuna-Score

```text
(best_window_score + final_window_score) / 2
```

bewertet die Trainingskurve. Greedy-Eval und Trainingsaufwand werden zwar
gespeichert, bestimmen aber nicht direkt den Gewinner.

Künftig gelten:

- `g`: mittlerer Gym-Score der Policy bei `epsilon = 0`
- `t`: hardwareunabhängige Approximation des Trainingsaufwands
- `o`: von Optuna maximierter Objective-Score

Nach jedem Trial wird die Policy über 20 Greedy-Episoden evaluiert:

```text
g = mean(eval_episode_returns)
```

Die Eval-Seeds sind fest und unabhängig vom Trainings-Seed. Dadurch bleiben
Trials vergleichbar, ohne die Laufzeit stark zu erhöhen.

**Referenzmessung in S0**

S0 liefert die Referenzwerte für `t`. Die Baseline wird mit mehreren
Trainings-Seeds ausgeführt. Pro Lauf werden gezählt:

- `env_steps`: ausgeführte Environment-Schritte
- `optimizer_updates`: ausgeführte Optimierungsschritte
- `processed_samples = optimizer_updates * batch_size`
- `wall_time_seconds`: nur zur Diagnose

Die Mittelwerte werden als Study-Attribute gespeichert:

```text
baseline_env_steps
baseline_processed_samples
```

**Trainingsaufwand**

Eine neue Funktion `training_effort(...)` berechnet:

```text
t =
    alpha * env_steps / baseline_env_steps
    + (1 - alpha) * processed_samples / baseline_processed_samples
```

Environment-Schritte approximieren Simulations- und CPU-Aufwand, verarbeitete
Samples den Lern- und GPU-Aufwand. Für einen typischen S0-Lauf gilt ungefähr
`t = 1`; `t = 0.8` bedeutet etwa 20 % weniger approximierten Aufwand.
Startwert: `alpha = 0.5`.

Wanduhrzeit bleibt gespeichert, fließt aber nicht in `t` ein. Der Score hängt
damit nicht von L4-Taktung oder Colab-Auslastung ab.

**Optuna-Score**

```text
quality = (g - 200) / (250 - 200)
o = quality_weight * quality - (1 - quality_weight) * (t - 1)
```

Startwert: `quality_weight = 0.9`. Die Landequalität dominiert; der Aufwand
entscheidet vor allem zwischen ähnlich guten Policies.

Einstellschrauben:

- `alpha`: Simulation gegenüber Lernen
- `quality_weight`: Qualität gegenüber Aufwand
- Qualitätsmarken `200` und `250`
- Anzahl und Seeds der Greedy-Episoden

**Änderungen an der Implementierung**

- `VectorTrainer` zählt `env_steps` und `optimizer_updates`.
- Das Trainingsergebnis stellt beide Zähler bereit.
- `training_effort(...)` berechnet `t`.
- `objective()` berechnet `g`, `t` und daraus `o`.
- S0 persistiert die Referenzwerte; S1 bis S4 verwenden sie.
- Die robuste Auswahl vergleicht weiterhin `trial.value`, nun also `o`.

Pro Trial werden `gym_score`, die beiden Zähler, `processed_samples`,
`training_effort`, `objective_score` und `wall_time_seconds` gespeichert.
Die Trainingskurve bleibt diagnostisch erhalten; Window-Scores bestimmen den
Gewinner nicht mehr.

### Warum Quality-Effort Score?

Der Name bezeichnet direkt die beiden Bestandteile: Landequalität `g` und
Trainingsaufwand `t`. Damit entspricht das Scoring besser dem Ziel der Quest:
mehr Greedy-Eval-Score bei möglichst geringem Aufwand. Beide Größen bleiben
getrennt messbar, die Gewichtung ist verständlich und das Ergebnis leicht
erklärbar. Bewusst entfallen zunächst nichtlineare Nutzenfunktionen,
Unsicherheitsabschläge und automatische Pareto-Auswahl. Falls die einfache
Metrik später nicht reicht, können robuste Gym-Scores über weitere Seeds,
ein Pareto-Diagramm oder aufgabenspezifische Qualitätsgrenzen ergänzt werden.
Die gespeicherten Rohwerte ermöglichen dies ohne erneutes Training.

## 2. S5 – Wirtschaftliches Trainingsbudget

S5 untersucht, wie viele Trainingsepisoden sich für die S4-Gewinnerparameter
wirtschaftlich lohnen. Die bisherigen Daten sprechen gegen eine vorschnelle
Kürzung: Episoden 501 bis 600 brachten häufig noch deutliche Verbesserungen.
Auch mehr als 600 Episoden könnten sich auszahlen.

S5 ist keine breite HPO-Studie. Die S4-Hyperparameter bleiben fest; nur das
Trainingsbudget wird variiert:

```text
num_episodes = [500, 600, 700, 800]
```

Jede Variante wird mit denselben wenigen Trainings-Seeds ausgeführt. Nach jedem
Training folgen 20 Greedy-Eval-Episoden. Erfasst werden `g`, `t`, `o` und die
Wanduhrzeit zur Kontrolle.

Die Ergebnisse werden als `g` gegen `t` dargestellt. Gesucht wird der kleinste
Trainingsaufwand, ab dem zusätzliche Episoden nur noch wenig Qualitätsgewinn
bringen. S5 darf deshalb auch ein Budget oberhalb von 600 Episoden wählen.

Das Ziel ist nicht das kürzeste Training, sondern der wirtschaftlichste Punkt
auf der Qualitäts-Aufwands-Kurve.

KISS:

- Keine neuen Trainer-Modi oder SearchSpace-Abstraktionen.
- Kein separater 100-Episoden-Abschlussvergleich.
- Kein Robustheitsabschlag und keine Pareto-Auswahl im Score.
- Rohwerte speichern; weitere Analysen später ohne neues Training ergänzen.
