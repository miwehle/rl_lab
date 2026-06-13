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
from hpo.lunar_lander.objective import create_objective

objective = create_objective(
    num_episodes=500,
    output_dir=HPO_OUTPUT_DIR,
    device=device,
)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

## Local Tests

Run from the repository root:

```powershell
dqn\.venv\Scripts\python.exe -m pytest hpo\tests
```
