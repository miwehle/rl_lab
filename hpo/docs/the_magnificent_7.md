# The Magnificent 7

Die HPO fuer Elise lebt nicht von einem einzelnen Trick, sondern vom Zusammenspiel sieben starker Rollen.

## 1. Human

Erkennt Muster, stellt Hypothesen auf, entscheidet die naechste Richtung.

## 2. StudyRunner

Orchestriert die Study-Serie: Study laden oder anlegen, Optuna laufen lassen, Checkpoint-Robustness auswerten, Incumbent speichern, Abschluss sichtbar machen.

## 3. Dashboard

Macht die laufende Studie visuell lesbar: Training, Optimization, Robustness. Es zeigt die Geschichte der Study live, damit der Mensch Muster, Fehler und Verbesserungspotenzial sieht.

## 4. Optuna

Schlaegt die naechste Hyperparameter-Kombination vor, aus Suchraum und Study-Historie.

## 5. VectorTrainer

Sammelt Transitions, fuellt Replay Memory, sampled Replay-Batches und orchestriert das Training in vielen Vector-Environments.

## 6. DQN

Ist das lernende Modell: Forward-Paesse, Q-Werte, Policy durch `argmax`.

## 7. PyTorch Training Stack

Rechnet die numerische Lernmechanik: Loss, Autograd, Gradienten und `AdamW`-Step.

## Mental Model

`VectorTrainer`, `DQN` und PyTorch Training Stack sind eng gekoppelt, aber nicht dasselbe.

Der `VectorTrainer` ist der Fluglehrer. Das `DQN` ist das Gehirn. PyTorch ist die Rechenmaschine, die das Gehirn anhand des Loss veraendert. Hinzu kommt der StudyRunner als Ablaufmanager.

Als Landkarte:  
Ein Mensch mit Dashboard zum Reinkucken in den HPO-Prozess,  
ein Manager, ein Trainer  und ein KI-Modell als Flugschueler.  
Technische Basis: PyTorch  und Optuna.
