# Verbesserungsideen

## Schwache Trainingsläufe früh beenden

**Intuitiv:** Wenn ein Lander nach etwa 300 Episoden noch deutlich unter einem Score von 100 liegt, ist es unwahrscheinlich, dass er durch bloßes Weitertrainieren zu einem Spitzenmodell wird. Der Versuch kann beendet und die Rechenzeit aussichtsreicheren Trials gegeben werden.

**Technisch:** Ein *Pruner* bewertet einen geglätteten Trainings- oder Zwischenevaluationsscore nach 300 Episoden. Liegt er unter einem zunächst empirisch gewählten Grenzwert, beendet Optuna den Trial.

## Das beste Modell eines Trainingslaufs sichern

**Intuitiv:** Ein DQN kann zunächst sehr gut werden und später wieder schlechter. Nur die finalen Hyperparameter zu speichern verliert das tatsächlich beste gefahrene Modell.

**Technisch:** Während des Trainings wird bei einem neuen besten Zwischenscore ein Checkpoint der Netzgewichte gespeichert. Die spätere Bewertung verwendet diesen Checkpoint statt grundsätzlich den Zustand der letzten Episode.

## Gute Trainingsläufe länger fahren lassen

**Intuitiv:** Zeigt ein Lander nach 500 Episoden noch einen klaren Aufwärtstrend, kann zusätzliches Training lohnend sein. Schlechte Läufe werden kurz gehalten, vielversprechende dürfen beispielsweise bis Episode 1000 weiterlernen.

**Technisch:** Eine einfache *Fortsetzungsregel* verwendet Niveau und Steigung des letzten Return-Fensters. Nur bei ausreichendem Score und positivem Trend wird das Episodenbudget erhöht.

## Den besten Trainingszeitpunkt robust bewerten

**Intuitiv:** Entscheidend ist nicht, wie gut der Lander zufällig am Ende war, sondern welches während des Trainings erreichte Modell in allen fünf Welten zuverlässig fliegt.

**Technisch:** Die besten Checkpoints beziehungsweise Trainingsfenster werden mit mehreren festen Evaluationsseeds über alle fünf Welten geprüft. Der mittlere Gym-Score dieser robusten Evaluation bestimmt den Gewinner.
