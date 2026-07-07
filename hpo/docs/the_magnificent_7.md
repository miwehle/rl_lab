# The Magnificent 7

Die HPO fuer Elise lebt nicht von einem einzelnen Trick, sondern vom Zusammenspiel sieben starker Rollen.

## 1. Mensch

Der Forscher.

Erkennt Muster, stellt Hypothesen auf, entscheidet die naechste Richtung.

## 2. Dashboard

Das Fenster in den HPO-Prozess.

Macht die laufende Studie visuell lesbar: Training, Optimization, Robustness. Es zeigt die Geschichte der Study live, damit der Mensch Muster, Fehler und Verbesserungspotenzial sieht.

## 3. StudyRunner

Der Dirigent der Studie.

Orchestriert die Study: laden oder anlegen, Optuna laufen lassen, Checkpoint-Robustness auswerten, Incumbent speichern, Abschluss sichtbar machen.

## 4. Optuna

Der Sucher im Hyperparameter-Raum.

Nutzt Suchraum und Study-Historie, um die naechste Hyperparameter-Kombination vorzuschlagen.

## 5. VectorTrainer

Der Flugtrainer.

Sammelt Transitions, fuellt Replay Memory, sampled Replay-Batches und orchestriert das Training in vielen Vector-Environments.

## 6. DQN

Der Flugschueler.

Ist das lernende Modell: Vorwärts-Durchläufe, Q-Werte, Policy durch `argmax`.

## 7. PyTorch

Die numerische Lernbasis.

Stellt bereit: Modell-Bausteine, Tensor/GPU-Rechnung, Autograd fuer Backprop und Optimizer wie `AdamW` fuer den Parameter-Update.

## Honorable Mention: Gymnasium

Gymnasium stellt die Umgebung (Environment) bereit, in welcher der Flugschueler Erfahrungen sammelt. Es ist die Welt, in der die KI lernt (Simulation).

## Mental Model

Die HPO ist ein Lernkreislauf: Optuna schlaegt eine Abstimmung vor, der StudyRunner bringt sie in den Lauf, der VectorTrainer laesst den Flugschueler in Gymnasium-Welten ueben, PyTorch passt das Modell an, und das Dashboard zeigt dem Menschen, was dabei passiert.

Der Mensch schaut auf das Ergebnis, kann den Suchraum anpassen und eine weitere Iteration starten.

Technische Basis: PyTorch, Optuna und Gymnasium.