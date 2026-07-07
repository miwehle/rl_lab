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

Schlaegt die naechste Hyperparameter-Kombination vor, aus Suchraum und Study-Historie.

## 5. VectorTrainer

Der Flugtrainer.

Sammelt Transitions, fuellt Replay Memory, sampled Replay-Batches und orchestriert das Training in vielen Vector-Environments.

## 6. DQN

Der Flugschueler.

Ist das lernende Modell: Vorwärts-Durchläufe, Q-Werte, Policy durch `argmax`.

## 7. PyTorch

Die numerische Lernbasis.

Stellt bereit: Modell-Bausteine, Tensor/GPU-Rechnung, Autograd fuer Backprop und Optimizer wie `AdamW` fuer den Parameter-Update.

## Mental Model

`VectorTrainer`, `DQN` und PyTorch sind eng gekoppelt, aber nicht dasselbe.

Der `VectorTrainer` ist der Flugtrainer. Das `DQN` ist das Gehirn. PyTorch ist die Rechenmaschine, die das Gehirn anhand des Loss veraendert. Hinzu kommt der StudyRunner als Dirigent.

Als Landkarte:  
Ein Mensch mit Dashboard zum "Reinkucken" in den HPO-Prozess,  
ein Dirigent, ein Trainer und ein KI-Modell als Flugschueler.  
Technische Basis: PyTorch und Optuna.
