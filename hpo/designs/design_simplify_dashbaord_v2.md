
# Vereinfachung des Dashboards

Der Nutzen des Study Series Plots trägt seine eigene mentale Last nicht.

Also:
- study_series entfernen
- neue Anordnung der Subplots: nur noch 2 Zeilen
- Zeile 1: immer nebeneinander: current_hps | current_study
- Zeile 2:
  - während Studie läuft: training_progress (trials)
  - final: robustness
- beide Zeilen sollen 245px Plot-Höhe haben
- die erste Zeile behält damit exakt ihre bisherige Höhe, weil die HP-Tabelle in dieser Höhe gut passt

Farben der Zeilen in current_hps bleiben unverändert.

Keine neue Farbsemantik einführen. Insbesondere keinen neuen Datenfluss über `trial.distributions`, `single()` oder zusätzliche Reporting-Felder einbauen.

`study_series.py` wird entfernt, nicht nur ungenutzt gelassen.

Der Checkpoint-Robustness-Plot wird für die ganze Zeilenbreite horizontal gezeichnet: y-Achse = Checkpoints/Kandidaten, x-Achse = Score, mit horizontalen Intervallen für `min..max`, `q05..q95`, `q25..q75` und Mean-Marker darüber. Keine Heatmap im ersten Schritt.

## Implementierungsmaxime

Nur entfernen und umverdrahten. Keine neue Bedeutung, keine neue Pipeline, keine Fallback-Matrix.

Diese Vereinfachung soll Produktions-LOC sparen. Vor und nach der Umsetzung `git diff --shortstat -- hpo/src` prüfen. Wenn der Produktions-Diff netto wächst, stoppen und neu abstimmen.

Nicht Teil von v2:
- neue HP-Farbsemantik
- `trial.distributions`
- zusätzliche Reporting-Felder
- neue Abstraktionen für spätere Layouts
