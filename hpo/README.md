# HPO

Hyperparameter optimization for the `dqn` project.

This project is the Optuna/Colab layer around `dqn`. The DQN package remains the
training library; HPO owns study setup, trial configuration, scoring, and Colab
workflow.

## Initial Setup

```powershell
cd rl_lab
dqn\.venv\Scripts\python.exe -m pip install -r hpo\requirements.txt
```

## Notebook

Use `hpo/notebooks/HPO_LunarLander.ipynb` to run optimization on Colab.

Recommended Colab runtime:

```text
L4 GPU
```

## Objective

The Optuna objective is created in code and then passed to a study:

```python
from hpo.evaluation.pruning import PruningConfig
from hpo.lunar_lander.objective import create_objective

pruning_config = None
# pruning_config = PruningConfig(start_episode=250, min_score=100.0)

objective = create_objective(
    num_episodes=500,
    output_dir=HPO_RUN_DIR,
    device=device,
    pruning_config=pruning_config,
)

study_db_path = HPO_STUDY_DIR / "lunar_lander_dqn.db"
study = optuna.create_study(
    study_name="lunar_lander_dqn",
    direction="maximize",
    storage=f"sqlite:///{study_db_path}",
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)
```

## Local Tests

Run from the repository root:

```powershell
dqn\.venv\Scripts\python.exe -m pytest hpo\tests
```
