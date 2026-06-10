# DQN Design Notes

This package is a small study implementation of DQN. The code is the source of
truth; this document only records the design intent that is useful to keep in
mind while changing it.

## Origin

This package started as a refactoring of the official
[PyTorch Reinforcement Learning (DQN) tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). The original notebook version is kept in
`notebooks/archive/Reinforcement_Learning_(DQN)_legacy.ipynb`.

Some CartPole-specific tutorial names were generalized during the refactoring.
For example, `episode_durations` became `episode_returns`, because the trainer
should also make sense for environments where an episode score is not simply a
duration.

## Goal

The DQN code should be reusable across simple Gymnasium environments. The trainer
receives an environment object and should not know whether it is training on
CartPole, MyFreeway, or another compatible environment.

Supported environments are intentionally limited for now:

```text
Discrete action space
Flat vector observations
Gymnasium reset/step API
```

The trainer only relies on the standard Gymnasium interaction shape:

```python
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
```

This keeps the implementation focused on the DQN learning loop instead of
environment-specific wiring, image observations, or CNN models.

## Training Boundary

The reusable boundary is the Gymnasium environment object:

```python
trainer = Trainer(env, seed=42)
result = trainer.train(config)
```

`Trainer` owns the long-lived training state: networks, optimizer, replay memory,
device, step counter, and environment. `TrainingConfig` describes one training
run. Calling `train()` again continues from the current trainer state.

The training loop should remain easy to read. The important DQN steps should stay
visible in `Trainer.train()`:

```text
select action
step environment
store transition
optimize q-network when appropriate
soft-update target network
finish episode bookkeeping
```

Helper methods are fine when they reduce noise, but they should not hide the
core learning flow.

## Mental Model for Usage

**`Trainer` -> `trainer.train(config)` -> `TrainingResult`**

`Trainer` owns the long-lived agent state: environment, device, networks,
optimizer, replay memory, seed, and step count. `TrainingConfig` contains the
choices for one training run. `trainer.train(config)` runs the DQN loop and can
be called again to continue from the current trainer state. `TrainingResult`
contains the trained Q-network and the episode metrics collected during that
run.

## Tuned Trainer

`TunedTrainer` is a small subclass for practical training improvements while
keeping the baseline trainer simple. Its current responsibilities are:

```text
wait for learning_starts before optimizing
optimize only every optimize_every steps
save the best checkpoint at the end of an episode
```

The subclass should stay small. If a tuning feature requires copying the whole
training loop, the base trainer probably needs a clearer extension point first.

## Notebook Role

Notebooks are consumers of the package. They may choose an environment, configure
training, plot results, and render episodes. They should not contain the
canonical model, replay memory, optimization step, or training loop.

## Possible Next Steps

```text
hyperparameter sweeps
```

## Non-Goals For Now

```text
CNN support
image observations
vectorized environments
advanced logging
prioritized replay
polished custom-environment rendering
```
