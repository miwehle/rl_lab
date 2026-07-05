# Integrated Checkpoint Robustness Eval

## Ziel

Das Dashboard soll Checkpoint Robustness Evaluation als kompaktes Overview-Panel integrieren.

Erste Implementierungsstufe: Das bisherige HP-Robustness-Panel wird durch Checkpoint Robustness ersetzt, weil fuer uns der konkrete gespeicherte Pilot wichtiger ist als die HP-Robustheit.

Die bestehende HP-Robustness-Implementierung bleibt erhalten; sie tritt nur im Dashboard-Overview vorerst in den Hintergrund.

Spaeter kann das Panel optional zwischen HP Robustness und Checkpoint Robustness umschaltbar werden, je nach Optimierungsschwerpunkt.

## Dashboard Story

Das Dashboard bleibt der HPO-Geschichtenerzaehler:

```text
Study Series                 Wo steht die Serie?
Current HPs                  Welche Parameter gelten gerade?
Study                        Was passiert in der aktuellen Study?
Checkpoint Robustness        Welcher konkrete Checkpoint ist robust?
Current Trial Training       Wie lernt der laufende Trial?
```

Checkpoint Robustness erzaehlt nicht zuerst einzelne Welten, sondern die Kandidatenentscheidung: Ist der nominell beste Checkpoint wirklich der beste konkrete Pilot, oder ist ein anderer Kandidat robuster?

## Plot im Overview

Im Dashboard-Overview zeigt ICRE einen Quantile-/Interval-Plot ueber die Checkpoint-Kandidaten.

Y-Achse:

```text
C1 trial ...
C2 trial ...
C3 trial ...
```

X-Achse:

```text
Gym score
```

Pro Kandidat werden gezeigt:

- `min..max`
- `q05..q95`
- `q25..q75`
- `median`
- `mean`

Die Quantile werden ueber die gesammelten Evaluationsscores des Kandidaten gebildet. Fuer den Overview reicht die Kandidatenebene; Welt-Details kommen spaeter in eine Detailansicht.

## Siegerregel

KISS-Regel: Der Checkpoint mit dem hoechsten Mean Gym Score gewinnt.

Alle weiteren Werte werden berichtet und im Plot sichtbar gemacht, gehen aber in dieser Implementierungsstufe nicht in die Auswahl ein:

- `median`
- `min..max`
- `q05..q95`
- `q25..q75`

Damit bleibt die Auswahl konsistent mit dem bisherigen Gym-Score-Ziel und vermeidet eine kuenstliche gewichtete Formel.

## Evaluation

Default fuer die erste Implementierungsstufe:

```text
top_n = 3
eval_episodes = bestehender Notebook-/Call-Wert
```

Das Dashboard-Panel wird auf Kandidatenebene aktualisiert: nach Kandidat 1, nach Kandidat 2 und nach Kandidat 3.

Es gibt in dieser Implementierungsstufe keine Zwischenupdates innerhalb eines Kandidaten. Das passt zum bestehenden `evaluate_checkpoint_robustness(...)`, bleibt uebersichtlich und vermeidet neue Progress-Datenstrukturen.

KISS-Vorteil: Die bestehende Implementierung berichtet bereits auf Kandidatenebene; dadurch braucht diese Implementierungsstufe kaum Aenderungen an `evaluate_checkpoint_robustness(...)`.

Falls ein Kandidat spaeter messbar lange dauert, kann granularerer Fortschritt als eigener Ausbauschritt folgen.

## Datenfluss

Der Overview braucht pro Kandidat eine kleine Summary.

Moegliche Datenstruktur:

```python
@dataclass(frozen=True)
class CheckpointRobustnessCandidate:
    label: str
    trial_number: int
    checkpoint_path: str
    scores: list[float]
```

Alternativ kann die erste Implementierung aus den bestehenden `RobustnessProgress.candidate_seed_scores` starten, wenn daraus genug Werte fuer die Quantile vorliegen.

Wichtig ist: Das Dashboard braucht fuer den Overview nicht die per-Welt-Rohscores.

## API-Skizze

Der bestehende Reporter-Pfad kann weiter genutzt werden:

```python
evaluate_checkpoint_robustness(
    ...,
    top_n=3,
    progress_fn=runner.reporter.report_robustness_evaluation,
)
```

ICRE nutzt direkt denselben Reporter-Kanal wie die bisherige HP Robustness; es braucht keine neue Orchestrierung, sondern vor allem eine andere Dashboard-Darstellung.

Objective-Hooks werden nur indirekt genutzt: Sie speichern vorher die eval-best Checkpoints und Trial-Attrs wie `evaluation_checkpoint_path`, die Checkpoint Robustness danach als Kandidaten einliest.

Das Dashboard erkennt am `RobustnessProgress.title == "Checkpoint Robustness Evaluation"`, dass das Panel als Checkpoint-Robustness-Overview gerendert wird.

## Umsetzungsschritte

1. Dashboard-Panel-Titel von HP Robustness auf einen progress-abhaengigen Titel umstellen.
2. HP-Robustness-Panel im Overview durch Checkpoint-Robustness-Darstellung ersetzen, sobald Checkpoint-Robustness-Progress vorliegt.
3. Quantile-/Interval-Plot fuer Kandidaten implementieren.
4. `evaluate_checkpoint_robustness(...)` im Notebook standardmaessig mit `top_n=3` verwenden.
5. Tests fuer Titel, Traces und Kandidatenquantile ergaenzen.

Der Panel-Titel darf nicht mehr fest `HP Robustness Evaluation` heissen, wenn im Panel tatsaechlich Checkpoint Robustness angezeigt wird; er soll die aktuell erzaehlte Robustness-Story korrekt benennen, zum Beispiel `Checkpoint Robustness Evaluation - Candidate 2/3`.

## Nichtziele

- Keine Detailansicht.
- Kein Zoom-in.
- Keine Heatmap.
- Keine per-Welt-Quantile im Dashboard-Overview.
- Keine neue Colab-Interaktivitaet.
- Keine Vermischung mit HP Robustness in dieser Implementierungsstufe.
