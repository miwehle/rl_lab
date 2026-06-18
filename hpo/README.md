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

The objective maximizes greedy Gym quality while penalizing training effort.
Use `run_study(...)` from the notebook; it persists the scoring configuration
and the S0 effort baseline in Optuna's SQLite storage.

```python
study = run_study(
    study_name="s1_qe_update_economy",
    search_space=SearchSpace1(),
    n_trials=40,
    num_episodes=600,
    baseline_env_steps=study0.user_attrs["baseline_env_steps"],
    baseline_processed_samples=study0.user_attrs["baseline_processed_samples"],
    study_dir=HPO_STUDY_DIR,
    device=device,
)
```

## Local Tests

Run from the repository root:

```powershell
dqn\.venv\Scripts\python.exe -m pytest hpo\tests
```
