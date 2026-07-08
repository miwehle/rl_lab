# Reward Shaping

Training-side reward shaping experiments for lander agents.

This project is separate from `hpo` on purpose. HPO owns hyperparameter optimization; `reward_shaping` owns small, explicit experiments that change the training reward and then judge the result with the real unshaped Gym score.

## Motto

Correct is what:

- robustly improves the Gym score and
- keeps the code simple.

## First Question

Can a small training penalty for ground side-thrust help Elise collect the normal LunarLander landing reward more reliably?

The target metric is the mean unshaped Gym score, in the same spirit as checkpoint robustness evaluation. Diagnostic metrics such as landed-but-truncated episodes can explain failures, but they are not the objective.

## Local Tests

Run from the repository root:

```powershell
dqn\.venv\Scripts\python.exe -m pytest reward_shaping\tests
```
