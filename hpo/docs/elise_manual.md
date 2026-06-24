# Elise Manual

*Elise, the simple and light planet lander.*

## Normal Operation

## Checkpointing

Checkpointing keeps the best model found during a trial, instead of only keeping the final model. It is optional. How you can plug it into Optuna objective:

```python
from hpo import checkpointing
from hpo.study import StudyRunner

OBJECTIVE_CFG = ObjectiveConfig(
    environment_factory=ENVIRONMENT_FACTORY,
    hooks=checkpointing.ObjectiveHookFactory(
        checkpoint_dir=STUDY_DIR / f"{RUN_NAME}_checkpoints",
        window=100), ...)

runner = StudyRunner(
    objective_cfg=OBJECTIVE_CFG, ...)
```

Then each trial writes one file, for example `trial_0003_best.pt`. If a better window appears later in the same trial, the file is overwritten.

## Resumability

## Dashboard
