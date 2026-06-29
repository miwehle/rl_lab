"""Hyperparameter optimization package for DQN lander agents.

Main characters:

- Use-case modules:
  - study: run one named Optuna study within a study series.
  - objective: run one Optuna trial.
  - hp_robustness: re-evaluate HP candidates with extra training seeds.
  - checkpoint_robustness: re-evaluate saved checkpoints as concrete pilots.
- View:
  - evaluation.dashboard: notebook dashboard for the running HPO.
- Reporting port:
  - study_reporting: reporter protocol and progress DTOs.
- Infrastructure:
  - checkpointing: checkpoint recorders, loading/saving, and objective hooks.
"""
