# Dashboard DB Monitor

## Ziel

Das Dashboard soll seine HPO-Daten aus der Optuna-DB und den gespeicherten Artefakten lesen koennen, statt sie nur waehrend des Laufs durch Reporter/Hooks durchgereicht zu bekommen.

Kernthese:

```text
Das Dashboard wird vom Push-Reporter zum DB-lesenden Monitor.
```

## Motivation

Die aktuelle Reporter-Loesung funktioniert, und das Hook-Pattern hat die Durchreichung schon deutlich gebaendigt.

Trotzdem koppelt sie Dashboard und laufende HPO-Orchestrierung:

```text
Objective / Hooks / StudyRunner -> Reporter -> Dashboard
```

Ein DB-lesendes Dashboard haette drei Vorteile:

1. Ein zweites Notebook kann live monitoren, waehrend im ersten Notebook die HPO-Zelle laeuft.
2. Fertige oder abgebrochene Studies koennen ohne Re-Run wieder angezeigt werden.
3. Die HPO-Implementierung kann einfacher werden, weil weniger Dashboard-Daten durch interne Module gereicht werden muessen.

## Datenquellen

Primaere Quelle ist die Optuna-DB:

- Studiennamen
- Trials
- Trial States
- Trial Params
- Trial Values
- Trial User Attrs
- Study User Attrs

Artefakte bleiben Dateien:

- Checkpoints
- Checkpoint-Metadaten
- Videos
- optionale Checkpoint-Qualification-Ergebnisse

Die DB enthaelt die stabilen Verweise auf diese Artefakte, zum Beispiel Checkpoint-Pfade, Scores und Metadaten in User Attrs.

## Dashboard Story

Das Dashboard bleibt der HPO-Geschichtenerzaehler. Die Datenquelle aendert sich, nicht die Rolle:

```text
Study Series                 aus Study User Attrs und Study Summaries
Current HPs                  aus Incumbent oder laufendem Trial
Study                        aus Trials der aktuellen Study
HP Robustness                aus gespeicherten Robustness-Attrs
Checkpoint Qualification     aus Checkpoint-Robustness/Qualification-Attrs
Current Trial Training       optional aus persistiertem Live-Progress
```

## Live Monitoring

Das erste Ziel ist read-only Monitoring in einem zweiten Notebook.

KISS-Start:

```python
dashboard = Dashboard.from_storage(DB_PATH)
dashboard.show(study_name="s3_10d_better_space")
```

oder als freie Funktion:

```python
show_dashboard(DB_PATH, study_name="s3_10d_better_space")
```

Refresh kann zunaechst manuell passieren, indem dieselbe Zelle erneut ausgefuehrt wird.

Spaeter kann ein Timer/Widget periodisch neu lesen, wenn das in Colab stabil genug ist.

## Live Training Progress

Live-Training-Progress ist der schwierigste Teil, weil rohe Episode Returns bisher nicht vollstaendig in der Optuna-DB liegen.

KISS-Migration:

1. DB-Dashboard zeigt zunaechst Study Series, Current HPs, Study, Robustness und Checkpoint Qualification.
2. Current Trial Training bleibt im Push-Dashboard, solange keine persistierte Quelle existiert.
3. Wenn noetig, schreibt der Trainer spaeter throttled TrainingProgress-Snapshots in ein kleines JSON-Artefakt oder Trial User Attr.

Keine grosse Event-Streaming-Loesung im ersten Schritt.

## ICRE Anschluss

ICRE passt gut zur DB-Monitor-Architektur.

`evaluate_checkpoint_robustness(...)` speichert bereits `study.user_attrs["checkpoint_robustness"]`.

Eine spaetere Checkpoint Qualification kann zusaetzlich speichern:

```text
checkpoint path
episodes
world summary
optional scores artifact path
```

Dann kann das Dashboard die Pilot-Qualifikation aus DB/Artefakten rekonstruieren, ohne dass die HPO-Zelle aktiv Daten durchreicht.

## Umsetzungsschritte

1. Kleine Loader-Funktion bauen, die eine Optuna-Study aus `storage` und `study_name` liest.
2. Bestehende Figure-Bau-Funktionen so nutzen, dass sie mit aus der DB rekonstruierten Studies laufen.
3. Eine Notebook-Helferfunktion fuer read-only Anzeige einfuehren.
4. Gespeicherte Robustness-Attrs in Dashboard-Datenstrukturen uebersetzen.
5. ICRE-Daten aus gespeicherten Attrs/Artefakten einlesen.
6. Erst danach pruefen, welche Push-Reporter-Durchreichung vereinfacht werden kann.

## Nichtziele fuer den ersten Schritt

- Kein vollstaendiges Live-Streaming.
- Kein Schreibzugriff des Monitor-Notebooks.
- Keine aggressive Refaktorierung von `StudyRunner`.
- Keine Ablösung des bestehenden Push-Dashboards, bevor der DB-Monitor stabil ist.
- Keine grosse Artefaktverwaltung; Pfade und kleine JSON-Dateien reichen vorerst.
