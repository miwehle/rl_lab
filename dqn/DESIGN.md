# DQN Module Design

## Goal

Build a small reusable DQN package that can train on both:

```text
CartPole-v1
MyFreeway-v0
```

The training code should not know which concrete environment it is training on. It should only rely on the Gymnasium API:

```python
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
```

For the first version, supported environments should have:

```text
Discrete action space
Flat Box observation space with shape=(n,)
```

This keeps the implementation simple and avoids CNN-specific code.

## Proposed Package Structure

```text
dqn/
  notebooks/
    Reinforcement_Learning_(DQN).ipynb
    reinforcement_learning_(dqn).py

  src/
    dqn/
      __init__.py
      config.py
      model.py
      training.py
      visualize.py

      envs/
        __init__.py
        myfreeway.py

      scripts/
        train_cartpole.py
        train_myfreeway.py
```

This is intentionally compact. Smaller pieces such as replay memory, action selection, optimization, and the training loop all belong in `training.py` for now. They can be split out later if the project grows.

## Module Overview

### `config.py`

Contains the training configuration as a dataclass.

Expected contents:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 128
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay: int = 2500
    tau: float = 0.005
    learning_rate: float = 3e-4
    replay_memory_capacity: int = 10_000
    num_episodes: int = 50
    max_steps_per_episode: int | None = None
    seed: int | None = None
    device: torch.device | None = None
```

The config is passed explicitly into the training function:

```python
result = train(env, config)
```

### `model.py`

Contains the neural network.

Expected contents:

```text
DQN(nn.Module)
```

The first model is a feed-forward network for flat observation vectors. It should not contain any CartPole or MyFreeway logic.

### `training.py`

Contains the reusable DQN training implementation.

Expected contents:

```text
Transition
ReplayMemory
TrainingResult
select_action(...)
optimize_model(...)
soft_update(...)
train(env, config) -> TrainingResult
```

This is the central module. It should not call `gym.make("CartPole-v1")` and should not import `MyFreewayEnv`. The concrete environment is created outside and passed in.

### `visualize.py`

Contains optional plotting and rendering helpers.

Expected contents:

```text
EpisodePlotter
record_episode(...)
show_animation(...)
```

`EpisodePlotter` replaces the old notebook function `plot_durations(...)`. Its main method should be `plot_returns(...)`, using the more general name `returns`. The plotted value is usually the episode return, while the y-axis label can be chosen per environment, for example `Duration` for CartPole or `Return` for MyFreeway.

### `envs/myfreeway.py`

Contains the custom Gymnasium environment.

Expected contents:

```text
MyFreewayEnv(gym.Env)
```

The environment implements:

```text
reset(...)
step(action)
render()
close()
action_space
observation_space
metadata
```

### `envs/__init__.py`

Optionally registers `MyFreeway-v0` with Gymnasium:

```python
from gymnasium.envs.registration import register

register(
    id="MyFreeway-v0",
    entry_point="dqn.envs.myfreeway:MyFreewayEnv",
)
```

### `src/dqn/scripts/train_cartpole.py`

Creates the CartPole environment and passes it to the reusable trainer:

```python
import gymnasium as gym

from dqn.config import TrainingConfig
from dqn.training import train

env = gym.make("CartPole-v1")
config = TrainingConfig()
result = train(env, config)
```

### `src/dqn/scripts/train_myfreeway.py`

Creates the MyFreeway environment and passes it to the same trainer:

```python
import gymnasium as gym
import dqn.envs

from dqn.config import TrainingConfig
from dqn.training import train

env = gym.make("MyFreeway-v0")
config = TrainingConfig(num_episodes=500)
result = train(env, config)
```

## Notebook To Module Mapping

```text
Notebook code / concept                  Target module

import gymnasium as gym                  src/dqn/scripts/train_cartpole.py
env = gym.make("CartPole-v1")            src/dqn/scripts/train_cartpole.py

device selection                         config.py or training.py helper
seed setup                               training.py

BATCH_SIZE                               config.py: TrainingConfig.batch_size
GAMMA                                    config.py: TrainingConfig.gamma
EPS_START                                config.py: TrainingConfig.eps_start
EPS_END                                  config.py: TrainingConfig.eps_end
EPS_DECAY                                config.py: TrainingConfig.eps_decay
TAU                                      config.py: TrainingConfig.tau
LR                                       config.py: TrainingConfig.learning_rate

class DQN                              model.py

Transition                              training.py
ReplayMemory                            training.py

steps_done                              training.py, local training state
select_action(state)                    training.py: select_action(...)

policy_net setup                        training.py: train(...)
target_net setup                        training.py: train(...)
optimizer setup                         training.py: train(...)
memory setup                            training.py: train(...)

optimize_model()                        training.py: optimize_model(...)

soft target update inside loop          training.py: soft_update(...)

training loop from notebook cell 16     training.py: train(env, config)

episode_durations                       training.py: TrainingResult
episode_lengths                         training.py: TrainingResult

plot_durations(...)                     visualize.py: EpisodePlotter.plot_returns(...)
record_episode(...)                     visualize.py
matplotlib animation code               visualize.py or notebook-only

MyFreewayEnv                            envs/myfreeway.py
gym registration for MyFreeway-v0       envs/__init__.py
```

The notebook names are CartPole-specific. During extraction they should be generalized:

```text
episode_durations -> episode_returns
plot_durations(...) -> plot_returns(...)
```

## Training Function

The notebook currently has a top-level training loop. In the package this should become a function:

```python
def train(env, config: TrainingConfig, plotter=None) -> TrainingResult:
    ...
```

The function is responsible for:

```text
validating action_space and observation_space
setting seeds, if configured
creating policy_net and target_net
creating optimizer and replay memory
running episodes
selecting actions with epsilon-greedy exploration
storing transitions
optimizing the policy network
soft-updating the target network
collecting episode returns and episode lengths
passing episode returns to the plotter, if one was provided
returning TrainingResult
```

Expected result object:

```python
@dataclass
class TrainingResult:
    policy_net: DQN
    episode_returns: list[float]
    episode_lengths: list[int]
    device: torch.device
```

`TrainingResult` replaces the notebook globals needed after training, especially `policy_net`, the collected episode metrics, and the resolved device for inference visualization.

## Mental Model for Usage

**`TrainingConfig` → `train(...)` → `TrainingResult`**

`TrainingConfig` contains the choices for a training run. `train(...)` performs the reusable DQN training loop. `TrainingResult` replaces the notebook globals needed after training, especially the trained policy network, collected episode metrics, and resolved device.

## Environment Wiring

`training.py` is environment-agnostic:

```python
result = train(env, config)
```

CartPole wiring:

```python
env = gym.make("CartPole-v1")
result = train(env, config)
```

MyFreeway wiring:

```python
env = gym.make("MyFreeway-v0")
result = train(env, config)
```

The reusable boundary is the Gymnasium environment object.

## MyFreeway First Version

`MyFreewayEnv` should start simple.

Initial assumptions:

```text
action_space = gymnasium.spaces.Discrete(n)
observation_space = gymnasium.spaces.Box(shape=(k,), dtype=np.float32)
terminated = True on goal or crash
truncated = True on max_steps
```

Possible actions:

```text
stay
up
down
left
right
```

Possible observation features:

```text
player_x
player_y
lane_1_car_x
lane_1_car_speed
lane_2_car_x
lane_2_car_speed
...
```

Possible rewards:

```text
+goal_reward
-crash_penalty
-step_penalty
+progress_reward, optional
```

The observation should be a flat numeric vector, preferably normalized to stable ranges.

## Notebook Role After Refactoring

The notebook should become a consumer of the package:

```text
choose env
choose config
call train(env, config)
plot results
record/render episodes
```

It should no longer contain the canonical implementation of the DQN model, replay memory, optimization step, or training loop.

A Colab notebook can stay small and use three code cells: setup, training, visualization.

Setup:

```python
!git clone https://github.com/DEIN_NAME/rl_lab.git
%cd rl_lab

!pip install -r dqn/requirements.txt

import sys
sys.path.insert(0, "dqn/src")
```

Training:

```python
import gymnasium as gym

from dqn.config import TrainingConfig
from dqn.training import train
from dqn.visualize import EpisodePlotter

env = gym.make("CartPole-v1")

config = TrainingConfig(
    num_episodes=50,
    seed=42,
)

plotter = EpisodePlotter(y_label="Duration")
result = train(env, config, plotter=plotter)
```

Visualization:

```python
from dqn.visualize import record_episode, show_animation

frames = record_episode(
    make_env=lambda: gym.make("CartPole-v1", render_mode="rgb_array"),
    policy_net=result.policy_net,
    device=result.device,
)

show_animation(frames)
```

## Non-Goals For The First Implementation

```text
CNN support
image observations
checkpointing
vectorized environments
hyperparameter sweeps
advanced logging
Double DQN
prioritized replay
polished MyFreeway rendering
```

These can be added later once CartPole and the first MyFreeway version both run through the same `train(env, config)` path.
