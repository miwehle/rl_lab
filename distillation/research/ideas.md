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
