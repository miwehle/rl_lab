# DQN

This project is based on:  [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)


## Initial Setup

```powershell
cd rl_lab
python -m venv dqn\.venv
dqn\.venv\Scripts\python.exe -m pip install --upgrade pip
dqn\.venv\Scripts\python.exe -m pip install -r dqn\requirements.txt
```

## VS Code Testing Panel

Open this repository in VS Code. The workspace is configured to use:

```text
dqn\.venv\Scripts\python.exe
```

Tests can be run from the VS Code Testing panel.

## Notebooks

Use `dqn/notebooks/RL_DQN.ipynb` for normal work.

(A tutorial-style notebook is archived at `dqn/notebooks/archive/Reinforcement_Learning_(DQN)_legacy.ipynb` for historical reference. The code in `dqn` originated from this notebook.)

## Terminal Scripts (Optional)

```powershell
cd rl_lab
dqn\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "dqn\src"
python -m dqn.scripts.train_cartpole
```

## Gymnasium Environments

### CartPole

#### Observation

`CartPole-v1` returns an observation as a flat NumPy array with shape `(4,)`.

Array structure:

`[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`

#### Actions

`CartPole-v1` has `Discrete(2)` actions:

`0 = push cart left`

`1 = push cart right`

### LunarLander

[Gym doc](https://gymnasium.farama.org/environments/box2d/lunar_lander/?utm_source=chatgpt.com)

#### Observation

`LunarLander-v3` returns an observation as a flat NumPy array with shape `(8,)`.

Array structure:

`[x_position, y_position, x_velocity, y_velocity, angle, angular_velocity, left_leg_contact, right_leg_contact]`

`left_leg_contact` and `right_leg_contact` are contact flags, usually `0.0` or `1.0`.

#### Actions

`LunarLander-v3` has `Discrete(4)` actions:

`0 = do nothing`

`1 = fire left orientation engine`

`2 = fire main engine`

`3 = fire right orientation engine`
