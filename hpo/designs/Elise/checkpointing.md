# Checkpointing

## Automatisches Speichern des besten Checkpoints im Trial 

> Zweck: "Gutes Modell entsteht zufällig im Trial → wir verlieren es nicht mehr."

Hintergrund: DQN-Qualität schwankt selbst bei gleichen HPs (ist seed-abhängig).

Entwurf: s. checkpoint_recorder_sequence.puml