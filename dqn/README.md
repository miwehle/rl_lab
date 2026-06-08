# DQN

This project is based on:  [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)


## Local Setup

### VS Code Testing Panel

Open this repository in VS Code. The workspace is configured to use:

```text
dqn\.venv\Scripts\python.exe
```

Tests can be run from the VS Code Testing panel.

### Terminal Scripts (Optional)

```powershell
cd rl_lab
dqn\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "dqn\src"
python -m dqn.scripts.train_cartpole
```

### Fresh Clone (Initial Setup)

```powershell
cd rl_lab
python -m venv dqn\.venv
dqn\.venv\Scripts\python.exe -m pip install --upgrade pip
dqn\.venv\Scripts\python.exe -m pip install -r dqn\requirements.txt
```

## Notebooks

Use `dqn/notebooks/RL_(DQN).ipynb` for normal work. The longer tutorial-style notebook is archived at `dqn/notebooks/archive/Reinforcement_Learning_(DQN)_legacy.ipynb` as historical reference.
