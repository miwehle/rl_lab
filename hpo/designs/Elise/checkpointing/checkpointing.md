# Checkpointing

## Automatisches Speichern des besten Checkpoints im Trial 

> Zweck: "Gutes Modell entsteht zufällig im Trial → wir verlieren es nicht mehr."

Hintergrund: DQN-Qualität schwankt selbst bei gleichen HPs stark (ist seed-abhängig).

==Checkpoint Store== ist Infrastruktur, nicht Strategie (mögliche Strategien neben HPO: s. BI9).

Entwurf: s. checkpoint_recorder_sequence.puml
- Speichern mittels Hook in VectorTrainer.train
- Es gibt zwar schon dqn\src\dqn\checkpointing.py aber das passt hier nicht ganz.

Ultra-Kurzform:
- VT += after_episode
- VT <|-- CheckpointingVT (nutzt den Hook um Checkpointing aufzurufen)
- checkpointing.py += BestCheckpointRecorder