# HPO

Hyperparameter optimization for the `dqn` project.

This project is the Optuna/Colab layer around `dqn`. The DQN package remains the training library; HPO owns study setup, trial configuration, scoring, and Colab workflow.

## Initial Setup

```powershell
cd rl_lab
dqn\.venv\Scripts\python.exe -m pip install -r hpo\requirements.txt
```

## Notebook

Use `hpo/notebooks/HPO_LunarLander.ipynb` for LunarLander and `hpo/notebooks/HPO_SolarSystemLander.ipynb` for Study Series 2A/2B.

Recommended Colab runtime:

```text
L4 GPU
```

## Objective

The objective maximizes greedy Gym quality while penalizing training effort. Use `run_study(...)` from the notebook; it persists the scoring configuration and the S0 effort baseline in Optuna's SQLite storage.

```python
objective_cfg = ObjectiveConfig(
    environment_factory=environment_factory,
    num_envs=16,
    eval_episodes=20,
)

study = run_study(
    study_name="s1_qe_update_economy",
    suggest_parameter_values=suggest_s1_parameter_values,
    incumbent_params=baseline_params,
    n_trials=40,
    database_path=study_db,
    objective_cfg=objective_cfg,
)
```

## Local Tests

Run from the repository root:

```powershell
dqn\.venv\Scripts\python.exe -m pytest hpo\tests
```
