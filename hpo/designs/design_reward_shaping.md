# Reward Shaping In HPO

## Purpose

This design sketches a small HPO-integrated reward-shaping spike for SolarSystemLander Elise.

The question is whether a training-only penalty for ground side-thrust can teach Elise to stop wasting time after touchdown and improve the unshaped Gym score.

## Motivation

The separate `reward_shaping` project proved the domain core is tiny, while a standalone experiment harness costs a lot of code.

For this first hypothesis, HPO already has the useful machinery: `StudyRunner`, `VectorTrainer`, Optuna, checkpointing, robustness evaluation, dashboard, DB/log handling, and Colab storage.

`StudyRunner` is the central reason this can stay small: it already coordinates training, evaluation, checkpoint preservation, robustness checks, reporting, and storage.

So the HPO spike should reuse that machinery instead of rebuilding a parallel experiment harness.

## Small Design

The HPO implementation should have three small parts.

1. A focused reward wrapper in `hpo.solar_system_lander`.
2. A small `EnvFactory` hook that can wrap training envs.
3. A minimal notebook change in `hpo/notebooks/solar_system_lander/train_elise.ipynb`.

## Reward Wrapper

The useful core from `reward_shaping.ground_thrust_penalty` should be reused or ported.

The wrapper applies a penalty only during training when all of these conditions hold:

- the selected action is a side-thruster action,
- both lander legs have ground contact,
- the lander body is still awake.

The shaped reward is:

```text
shaped_reward = gym_reward - ground_thrust_penalty
```

Otherwise the shaped reward equals the original Gym reward.

The first implementation can live in a small module such as:

```text
hpo/src/hpo/solar_system_lander/reward_shaping.py
```

Keeping it separate from `environment.py` makes it easy to delete if the hypothesis fails.

## EnvFactory Hook

`EnvFactory` should get the smallest useful hook for training env construction.

Sketch:

```python
EnvFactory(
    OBSERVATION_MODE,
    world_mix=...,
    env_wrapper=lambda env: GroundThrustPenaltyEnv(env, ground_thrust_penalty=0.5),
)
```

The wrapper should be applied to training envs created by `make_training_env`.

Evaluation should stay unshaped. The existing `evaluation_envs()` should continue to return the original unshaped Gym reward path unless we deliberately decide otherwise.

## Notebook Use

In `train_elise.ipynb`, the experiment should be a normal HPO study.

The `objective-config` cell creates the shaped `ENV_FACTORY`, then `StudyRunner` runs as usual.

That gives us Optuna, dashboard, checkpointing, and checkpoint robustness without a separate harness.

## Metrics

Gym score remains king.

The main score should be the existing HPO objective and checkpoint robustness evaluation on unshaped environments.

For diagnosis, we may also count ground side-thrust steps before/after, but that diagnostic must not replace the Gym score.

## Implementation

### Approach
- Neues HPO-Modul für Reward Shaping anlegen, grob als Port/Kopie von `reward_shaping/src/reward_shaping/ground_thrust_penalty.py`.
- Darin liegt der eigentliche Wrapper: `RewardShapingEnv`, ein `gym.Wrapper`.
- `EnvFactory` bekommt diesen Wrapper bzw. eine passende Wrapper-Funktion als optionales Argument und speichert sie als Attribut.
- Wenn dieses Attribut gesetzt ist, wendet `make_training_env()` den Wrapper auf die Trainings-Envs an.
- Evaluation bleibt dadurch unshaped, solange `evaluation_envs()` unverändert bleibt.

### Notebook

Ob Elise mit oder ohne Reward Shaping trainiert, wird im Notebook bei der Erzeugung der `EnvFactory` entschieden. Ohne Wrapper trainiert Elise auf der normalen Umgebung; mit übergebenem Trainings-Wrapper trainiert sie auf der entsprechend geshapten Umgebung, während Evaluation und HPO-Score unshaped bleiben.


### Expected Code Size

The HPO spike should be much smaller than the standalone reward-shaping harness.

Expected rough size:

```text
reward wrapper and predicate: 30-45 prod LOC
EnvFactory hook:             10-20 prod LOC
notebook change:              5-15 notebook LOC
focused tests:               30-60 test LOC
```

The design target is a compact spike, not a generic reward-shaping framework.

## Boundary

Do not move the standalone harness into HPO.

Use HPO because the experiment machinery already exists there.

Keep the shaped reward training-only unless a later experiment explicitly needs shaped evaluation.

Correct is what robustly improves the Gym score and keeps the code simple.
