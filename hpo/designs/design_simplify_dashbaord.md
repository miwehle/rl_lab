
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

Farben der Zeilen in current_hps: Ist der HP
- nur in Baseline, dann: weiß
- in suggest_* aber als nur ein Wert, dann: hellgrau
- in suggest_* und in Optuna-Optimierung, dann: gelb

Die Farbgebung gilt bereits während die Studie läuft, nicht erst rückblickend.

Technische Festlegung: Nach `suggest_parameter_values(trial, incumbent_params)` sind im laufenden Optuna-Trial `trial.params` und `trial.distributions` verfügbar. `trial.distributions[name].single()` ist die maßgebliche Semantik für "nur ein Wert". Also: kein Eintrag in `trial.params`/Distributionen = Baseline/Incumbent (weiß), Distribution mit `single() == True` = `suggest_*` fixiert auf einen Wert (hellgrau), Distribution mit `single() == False` = echter Optuna-Suchraum (gelb).

`study_series.py` wird entfernt, nicht nur ungenutzt gelassen.

Der Checkpoint-Robustness-Plot wird für die ganze Zeilenbreite horizontal gezeichnet: y-Achse = Checkpoints/Kandidaten, x-Achse = Score, mit horizontalen Intervallen für `min..max`, `q05..q95`, `q25..q75` und Mean-Marker darüber. Keine Heatmap im ersten Schritt.
