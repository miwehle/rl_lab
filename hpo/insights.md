# HPO Insights

## Trainingsdurchsatz

Empirisch aus 103 Trials auf einer L4 mit 20 parallelen Environments:

\[
\text{Env-Steps/s} \approx 851
\left(\frac{\text{optimize\_every}}{2}\right)^{0.854}
\]

`optimize_every` erklärt nahezu die gesamte Variation des Durchsatzes. Größere Werte bedeuten weniger Optimizer-Updates und damit mehr Environment-Steps pro Sekunde.

| `optimize_every` | Gemessener Median Env-Steps/s |
|---:|---:|
| 2 | 858 |
| 4 | 1.483 |
| 8 | 2.666 |

Ein einfaches Kostenmodell schätzt etwa 0,15 ms für einen Environment-Step und 2,03 ms für Backpropagation plus Optimizer-Update. Ein Update dauert also rund 14-mal so lange wie ein Environment-Step.

Die vollständige Vier-HP-Regression erreicht etwa 1,1 % mittleren relativen Fehler, aber `batch_size`, `num_episodes` und `learning_starts` liefern nur geringe zusätzliche Erklärung. Die einfache Formel ist daher vorzuziehen.

## Modellqualität hängt stark vom Trainingsseed ab

Gleiche HPs erzeugen je nach Trainingsseed sehr unterschiedlich gute Modelle. In der Elise-Studienreihe erreichten einzelne Trials einen mittleren Score von etwa 167 über alle fünf Welten (auf der Erde: 237,5). Die robuste Neuauswertung derselben HPs erreichte dagegen nur 113 beziehungsweise 92. Gute HPs sind daher ein erster Filter, kein Ersatz für das Speichern eines konkret guten Modells.
