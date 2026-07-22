# Distillation Ideas

## Best Student Checkpoint Per Training Run

Beim RL-Training mit StudyRunner und VectorTrainer war selten das Modell am Ende eines Trials das beste. Der mean trailing score schwankte ueber die Episoden; gute Checkpoints lagen oft im letzten Drittel, aber nicht zwingend am Schluss.

Idee: Bei der Teacher-Student-Distillation nicht automatisch den letzten Epoch-Stand als besten kleinen Piloten behandeln. Stattdessen pro Run mindestens den besten Student-Checkpoint waehrend des Trainings sichern.

KISS-Start:

- `train_student(...)` speichert weiter den letzten Checkpoint.
- Zusaetzlich wird der beste Epoch-Checkpoint nach Validierungsmetrik gesichert, z. B. `val_loss`.
- Der Run soll klar ausweisen, welcher Checkpoint `last` und welcher `best` ist.

Wichtiges Risiko: `val_loss` oder Action-Agreement koennen besser werden, ohne dass die echte Flugqualitaet besser wird. Die bisherigen Runs zeigen bereits, dass niedrigere Distillation-Losses nicht automatisch hoehere Gym-Scores bedeuten.

Bessere spaetere Variante: alle paar Epochen eine kleine feste Gym-Evaluation ausfuehren und den besten Eval-Checkpoint sichern, analog zur HPO-Idee des besten Checkpoints im Trial.

## Hard-Seed Flight School

Die 64x64-Micro-Elise fliegt im Median schon gut, scheitert aber weiter an harten Ausreissern, besonders auf Venus und teilweise Earth. Mehr Venus insgesamt half zuletzt nicht; der Median blieb gut, aber der Mean wurde nicht besser.

Idee: nicht ganze Welten staerker reinmischen, sondern gezielt schwierige Welt/Seed-Paare als zusaetzliche Trainingsfaelle sammeln. Das waere chirurgischer als ein groesserer World-Mix.

Skizze:

```python
EXTRA_SEEDS_BY_WORLD = {
    World.VENUS: [20, 57, 87, 63, 31, 22] * 5,
    World.EARTH: [87, 57, 63, 31] * 3,
}
```

KISS-Start: `collect_teacher_dataset(...)` und die parallele Variante koennten optional `extra_seeds_by_world` annehmen und diese Episoden an die normale Episodenliste anhaengen. Das Training selbst muss davon nichts wissen.

Offene Frage: Wenn Micro-Elise diese Grenzfaelle mit ca. 5000 Parametern nicht stabil aufnehmen kann, ist das ein echtes Kapazitaetslimit oder nur eine Frage der richtigen Beispiele.

## Distillation StudyRunner

Die Distillation hat inzwischen mehrere echte Stellschrauben: `learning_rate`, `batch_size`, `epochs`, `epsilon`, World-Mix bzw. Trainingsgewichtung und spaeter Hard-Seed-Repeats. Manuelles Durchprobieren wird schnell unuebersichtlich.

Idee: Distillation als eigenes Projekt behalten, aber HPO/Optuna/StudyRunner als Infrastruktur nutzen. Der StudyRunner waere wieder der Freund fuer Trials, Persistenz, Vergleichbarkeit, bestes Ergebnis und spaeter ein Dashboard.

KISS-Start: teure Datasets vorab sammeln und pro Trial nur Training-Parameter optimieren. `epsilon` kann zunaechst als Auswahl zwischen vorgenerierten Datasets behandelt werden.

Moegliche Objective:

```text
weighted_eval_mean = heavy-world-weighted Gym evaluation score
```

Reason: Overall mean alleine kann eine gute Moon/Mars/Mercury-Elise belohnen, obwohl die Problemfaelle auf Venus/Earth liegen. Eine Objective sollte deshalb Venus/Earth staerker gewichten oder den schlechtesten Welt-Mean mitberuecksichtigen.

Spaeter: Dashboard fuer Trial-Vergleich, Score-Verteilungen pro Welt, Best-Checkpoint, Loss/Agreement-Verlauf und Failure-Seeds.
