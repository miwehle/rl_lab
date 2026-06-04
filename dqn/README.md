# DQN

## Local Setup

```powershell
cd rl_lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r dqn\requirements.txt
```

Run scripts from the repository root so `dqn` is importable as a package:

```powershell
python -m dqn.scripts.train_cartpole
```

## Colab Setup

```python
!git clone https://github.com/DEIN_NAME/rl_lab.git
%cd rl_lab
!pip install -r dqn/requirements.txt
```
