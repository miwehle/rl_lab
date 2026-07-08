# Reward Shaping Design

## Purpose

`reward_shaping` is a separate project for small training-side reward shaping experiments with lander agents.

==The first project question is whether Elise can be trained out of firing a side thruster after touchdown, because this can prevent the normal LunarLander terminal landing reward of `+100`.==

## Rough Design: The 4 Main Roles

The first implementation has four main roles.

1) ==`RewardShapingEnv` is the training environment wrapper.== It changes the reward returned by the environment before the transition enters replay memory.
2) ==`dqn.vector_training.VectorTrainer` is the training engine.== It brings the DQN training loop, replay memory, TD target, optimizer, and default DQN model path, so `reward_shaping` does not need a second DQN trainer.
3) ==The run directory is the file-system I/O boundary.== It stores input checkpoints, run metadata, training summaries, evaluation scores, and the shaped output checkpoint.
4) ==Unshaped evaluation is the truth check.== It evaluates checkpoints with the original Gym reward and decides whether the experiment improved mean Gym score.

## Details

### RewardShapingEnv

`reward_shaping.ground_side_thrust` contains the first focused shaping environment.

`RewardShapingEnv` applies a small training penalty when the agent fires a side thruster while both lander legs are on the ground and the lander is still awake.

The shaped reward is:

```text
shaped_reward = gym_reward - ground_side_thrust_penalty
```

The penalty applies only when all of these conditions hold:

- the selected action is a side-thruster action,
- both lander legs have ground contact,
- the lander body is still awake.

Otherwise the shaped reward equals the original Gym reward.

The shaped training reward is a learning aid, not the truth metric.

The class name stays easy to speak; the module and docstring carry the concrete ground-side-thrust details.

### Unshaped Evaluation

Unshaped evaluation uses two score measurements.

Diagnostics such as `both_contact + awake + side_thruster` tails and landed-but-truncated episodes can explain behavior, but they do not replace the Gym score.

#### Historical Scoring

==`historical_score` is the decisive comparison with the historical `253` checkpoint score.==

It uses the same protocol as that score: greedy policy, original Gym reward, `eval_seed=10000`, `10` episodes per world, all five worlds, and the mean over the world means.

#### Robust Scoring

`robust_score` is the stronger second measurement.

It uses the same protocol, but with more episodes per world. The first default should match the current HPO notebook checkpoint robustness setting: `50` episodes per world, with one seed per episode: `10000..10049`.

### Run Artifacts

Reward shaping runs should write file artifacts instead of an experiment database until the project has a clear need for database queries or resume semantics.

Use one directory per run:

```text
reward_shaping/runs/<run_id>/
  inputs/
    initial_checkpoint.pt
  outputs/
    config.yaml
    training_summary.yaml
    eval_scores.csv
    shaped_checkpoint.pt
```

`initial_checkpoint.pt` is a local copy of the checkpoint used as the starting point, so Colab runs do not need to fetch the same Drive artifact repeatedly.

`config.yaml` records the human-readable run configuration and input metadata.

`training_summary.yaml` records compact training outcomes.

`eval_scores.csv` records the unshaped Gym evaluation scores.

`shaped_checkpoint.pt` is the newly trained checkpoint produced by the run.

## Boundaries

HPO remains the separate project for hyperparameter optimization.

`reward_shaping` should not depend on `hpo`.

HPO may later call into `reward_shaping`, but that dependency should point from HPO to reward shaping, not the other way around.

The first implementation should be the smallest useful experiment, not a generic reward-shaping framework.

## Design Rules

Keep the shaped reward training-only.

Prefer one explicit environment wrapper over a configurable shaping system until a second current use case exists.

Keep diagnostics small and tied to decisions; do not add reporting fields just because they might be interesting later.

==Correct is what robustly improves the Gym score and keeps the code simple.==
