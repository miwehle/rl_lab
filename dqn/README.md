# DQN

This project is based on:  [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)


## Local Setup

```powershell
cd rl_lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r dqn\requirements.txt
$env:PYTHONPATH = "dqn\src"
```

Run scripts from the repository root:

```powershell
python -m dqn.scripts.train_cartpole
```

Run tests from the repository root:

```powershell
pytest -c dqn\pytest.ini dqn\tests
```

## Colab Setup

```python
!git clone https://github.com/DEIN_NAME/rl_lab.git
%cd rl_lab
!pip install -r dqn/requirements.txt
import sys
sys.path.insert(0, "dqn/src")
```
