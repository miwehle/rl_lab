# Checkpointing

## Automatisches Speichern des besten Checkpoints im Trial 

> Zweck: "Gutes Modell entsteht zufällig im Trial → wir verlieren es nicht mehr."

Hintergrund: DQN-Qualität schwankt selbst bei gleichen HPs stark (ist seed-abhängig).

==CheckpointStore== ist Infrastruktur, nicht Strategie (mögliche Strategien neben HPO: s. BI9).

Entwurf: s. checkpoint_recorder_sequence.puml