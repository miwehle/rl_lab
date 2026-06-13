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

## Local Tests

Run from the repository root:

```powershell
dqn\.venv\Scripts\python.exe -m pytest hpo\tests
```

