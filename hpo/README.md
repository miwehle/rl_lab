# HPO

Hyperparameter optimization for the `dqn` project.

This project is the Optuna/Colab layer around `dqn`. The DQN package remains the training library; HPO owns study setup, trial configuration, scoring, and Colab workflow.

## Initial Setup

```powershell
cd rl_lab
dqn\.venv\Scripts\python.exe -m pip install -r hpo\requirements.txt
```

## Notebook

Useful notebooks:

- `hpo/notebooks/lunar_lander/HPO_LunarLander.ipynb` for LunarLander.
- `hpo/notebooks/solar_system_lander/train_elise.ipynb` for SolarSystemLander HPO studies.
- `hpo/notebooks/solar_system_lander/integration_tests.ipynb` for SolarSystemLander integration smoke tests.
- `hpo/notebooks/solar_system_lander/videos.ipynb` for checkpoint landing videos.
- `hpo/notebooks/basics/basics.ipynb` for small notebook basics.

Recommended Colab runtime:

```text
L4 GPU
```

## Public API

HPO uses two public API levels to keep notebooks, package boundaries, and tests aligned.

`notebook-public` objects are objects directly used by the SolarSystemLander HPO notebooks. They are re-exported from `hpo/__init__.py`, so notebook code can treat `hpo` as the user-facing API surface.

`to-higher-level public` objects are lower-level package objects directly used by a higher-level public object. They are re-exported only from the `__init__.py` of their own lower-level package.

Direct tests should usually target only one of these two public API levels. Private helpers and incidental implementation details should usually be tested through their public users instead.

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
