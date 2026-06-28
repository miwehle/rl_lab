# HPO Package Definition

Purpose: this file is the compact startup context for future Codex sessions working on the `hpo` package.

Read this before making design or implementation changes in `hpo`, especially after a context reset.

The package orchestrates hyperparameter optimization for DQN lander agents, with Optuna studies, vectorized training, checkpointing, robust candidate re-evaluation, Colab persistence, and a notebook dashboard for the human running the HPO.

The package's first practical goal is to produce and preserve the best possible model checkpoints by evaluation Gym score.

The second practical goal is to grow the dashboard as the central HPO tool: it should make the optimization understandable to the human and may later become a more general deep-learning dashboard.

Related sequence diagrams:

- `hpo_study_flow.puml`: StudyRunner, dashboard, objective, trainer, and robust selection.
- `hpo_training_checkpointing.puml`: training progress, training checkpoints, evaluation-best checkpoints, and adaptive training extension.

## Mental Model

The core story is: a human starts a study series in a notebook, `StudyRunner` runs one named Optuna study at a time, each trial trains a DQN with `VectorTrainer`, the objective evaluates the trained model across one or more worlds, robust selection re-checks the best candidates with extra seeds, and the dashboard tells the live story of the study series.

A study series is the long-running HPO campaign; in this project, one SQLite database is assumed to represent one study series.

A study is one named step in a series, for example `s1_flight_hours` or `s2_exploration`.

A baseline is the incumbent at the start of a study series or resumed series: parameters plus score.

An incumbent is the current title holder for the series; it seeds future trials through `incumbent_params`, sets the checkpoint minimum score, and is written to study user attrs when a study finishes.

The current optimization is deliberately simple: `StudyRunner.run(...)` is still the main orchestration point; do not add a hook layer to `study.py` without a strong current need.

KISS rule for this package: prefer small orchestration helpers and explicit data flow over speculative framework code.

KISS does not mean avoiding good libraries or future reuse; it means avoiding complexity that has not earned its keep yet.

## Main Modules

`hpo/src/hpo/study.py` owns study-series orchestration.

Important objects in `study.py`: `Incumbent`, `StudyRunner`, `run_study`, `_with_checkpoint_min_score`, `_with_training_progress`, and `_study_already_finished`.

`StudyRunner.run(...)` loads or creates the current study, briefs the reporter with series context, runs Optuna until the target finished trial count, runs robust selection, updates incumbent attrs, syncs DB/log if configured, updates the dashboard, and appends the study to its in-memory series.

`run_study(...)` treats `n_trials` as target total finished trials, not additional trials; this is what makes Optuna resume safely after Colab reconnects.

If a study is already fully finished, `StudyRunner.run(...)` prints `Study already finished.` and returns before robust selection; the finished criterion is enough finished trials plus `robust_best_score` and `incumbent_score` attrs.

`hpo/src/hpo/objective.py` owns the Optuna objective.

The objective uses a Hook Object pattern through `ObjectiveConfig.hooks` so checkpointing and live training progress do not clutter `objective.py`.

`create_objective(...)` suggests params, builds `VectorTrainingConfig`, creates an `ObjectiveContext`, creates hooks for the trial, builds a trainer, trains, evaluates greedy Q-net performance over worlds, finalizes hooks, saves trial attrs, and returns the mean score.

`ObjectiveContext` is the small hook-relevant context object that is filled through the objective; hooks receive it at `for_trial(ctx)` and `finalize_trial(ctx, save)`.

`vector_training_config(...)` maps HPO hyperparameters to `VectorTrainingConfig`; HPO currently enables adaptive training extension with `adaptive_extension_window=50` and early stopping through `ObjectiveConfig.early_stopping_score`.

The objective saves `trained_episodes` so planned vs. actual training length is visible after adaptive extension or early stopping.

`hpo/src/hpo/checkpointing.py` owns training-time checkpoints and HPO objective hooks.

`ObjectiveHookFactory` can copy itself with `with_min_score(...)` and `with_training_progress(...)`; `study.py` uses small `_with_*` helpers to apply those hook-copy methods while keeping the objective config immutable.

`CheckpointingTrainer` subclasses `VectorTrainer` and records checkpoints after episodes before delegating to the base trainer's `_after_episode`.

`BestCheckpointRecorder` saves a checkpoint whenever the trailing return window beats the previous training checkpoint score and any configured minimum score.

Training checkpoint score is the trailing training mean, not the final evaluation score.

`EvaluationBestCheckpointRecorder` saves the evaluated Q-net after world evaluation when the evaluation score beats the configured minimum score; this protects high-score pilots that training-window checkpointing did not save.

`ObjectiveHooks.finalize_trial(ctx, save)` is the post-evaluation hook that saves training-checkpoint attrs and evaluation-checkpoint attrs.

`best_checkpoint(study)` scans both `trials` and `robustness` checkpoint subdirectories; it prefers `*_eval_best.pt` evaluation checkpoints, and only falls back to training-window `*_best.pt` checkpoints when no evaluation checkpoints exist.

`hpo/src/hpo/robust_selection.py` owns robust candidate re-evaluation.

`select_robust_best(...)` sorts complete trials by objective value, re-runs the top candidates with extra training seeds, reports progress, stores robustness checkpoint metadata when the objective saved it, and writes `robust_best_params`, `robust_best_score`, and `robustness_checkpoints`.

Robustness currently optimizes for mean score across candidate seeds; this favors reproducible producers, not necessarily the single highest-potential pilot.

The user may prefer "class over mass": a future selection policy might consider `max` or top-k mean across seeds, but this is not implemented yet.

`hpo/src/hpo/study_reporting.py` owns the reporting protocol and small reporting data classes.

`StudySeriesReporter` is implemented by the dashboard and has methods for series context, optimization progress, robustness progress, and live training progress.

`TrainingProgressPlotter` adapts the DQN trainer plotter protocol into `TrainingProgress` reports.

`hpo/src/hpo/evaluation/dashboard.py` owns the notebook dashboard and is a central HPO tool, not merely a reporting afterthought.

The dashboard is the visual interface between the human and the running HPO, and it should make the study series feel readable while it runs.

Dashboard panels: Study Series, Current HPs, current Study, HP Robustness Evaluation, and Current Trial Training.

Current Trial Training plots raw episode returns, trailing mean over the checkpoint window, epsilon on a secondary axis, and a horizontal reference line from the current best checkpoint score or initial checkpoint threshold.

For SolarSystemLander, Current Trial Training colors episode-return dots by env label/world while keeping the chronological return line thin and gray; labels are extracted in the HPO hook from `env.envs[*].world.name`, not hard-coded into `VectorTrainer`.

`Dashboard(render_mode="safe")` clears and redisplays the whole dashboard; this is robust in notebooks/Colab but can flicker.

Training progress updates are throttled by `training_update_interval_seconds=5.0` by default so live training does not redraw on every episode.

Only training progress is throttled; optimization and robustness updates render immediately.

`hpo/src/hpo/notebook/colab.py` owns Colab setup and storage helpers.

`prepare_storage(...)` restores DB/log from Drive to local runtime paths, configures logging, and returns a `Storage` whose `backup()` copies DB/log back to Drive.

Current notebook storage backup does not automatically save model checkpoints to Drive.

`hpo/src/hpo/notebook/optuna.py` owns small notebook helpers for inspecting Optuna study databases, plus `neighbors(...)` for local categorical search around an incumbent value.

## DQN Vector Trainer

`dqn/src/dqn/vector_training.py` is part of the DQN package, but it is central to HPO.

`VectorTrainer` collects experience from many Gymnasium environments at once and trains the DQN from a vector replay memory.

`VectorTrainingConfig` extends the base DQN training config with `learning_starts`, `optimize_every`, and `adaptive_extension_window`.

Adaptive training extension is implemented in `VectorTrainer.train(...)`, not in the dashboard.

Adaptive extension protects late-learning trials from being cut off at the initial episode target.

Early stopping protects HPO time from clearly weak trials: if the trailing mean over the adaptive extension window is below `early_stopping_score` at the halfway point, training stops and the objective returns that trailing mean as the trial score.

HPO defaults `ObjectiveConfig.early_stopping_score` to `-250.0`; set it to `None` in notebooks to disable early stopping.

The rule is evaluated only when the current target episode count has been reached.

The rule can extend training in blocks of half the original `num_episodes`, up to a hard maximum of `4 * num_episodes`.

The rule uses these deliberate variable names:

```python
lm50 = mean(last 50 episode returns)
pm50 = mean(previous 50 episode returns)
diff = lm50 - pm50
armstrong_factor = lm50 / 100
learning_momentum = diff * armstrong_factor
is_new_best_mean = lm50 > previous_best_mean
should_extend = learning_momentum > 10 and is_new_best_mean
```

The "Armstrong factor" intentionally favors candidates that are both still improving and already in a promising score region; `is_new_best_mean` prevents long score waves from extending unless the current trailing mean reaches a new trial-best level.

The user likes the German name "Adaptive Verlaengerung" for this feature; in English docs and code use "adaptive training extension".

## Dashboard Lessons

The live training plot already proved valuable: it showed trials whose trailing mean was still rising at the initial training cutoff.

This observation directly motivated adaptive training extension.

The live plot can expose raw outliers such as very negative LunarLander returns; Gym scores are cumulative rewards and can go far below the final crash penalty.

If the plot's y-axis is dominated by outliers, a future KISS improvement could use robust axis scaling while still making outliers visible.

Do not assume `Study Series` can reconstruct previous studies after Colab reconnect yet.

Currently, `StudyRunner.studies` is in-memory context; after reconnect it starts empty unless previous studies are reloaded and supplied by future code.

The desired future behavior is that a resumed study series shows previous studies from the same DB immediately in the Study Series panel.

## Checkpointing Lessons

Important lesson from the "211" run: training-checkpoint best is not the same thing as evaluation-score best.

The 211 score appeared in the robustness dashboard as an evaluation result, but no corresponding model checkpoint existed because checkpointing only saved based on trailing training mean.

The runtime can still live while a model from a completed objective call is already gone; if no checkpoint was saved, the local function's `result.q_net` is not recoverable later.

Evaluation-best checkpointing has been added so future Armstrong-level models should be saved after evaluation, not only when the training trailing mean looks best.

Current design: after each objective evaluation, `ObjectiveHooks.finalize_trial(...)` can save exactly the evaluated Q-net as an `*_eval_best.pt` checkpoint and write `evaluation_checkpoint_*` attrs.

Drive policy should stay intentional: local runtime may keep all trial and robustness checkpoints, but Google Drive should ideally receive the selected incumbent/evaluation-best checkpoint rather than every transient file.

For emergency preservation, copying the whole local checkpoint directory to Drive is acceptable when it is tiny.

## Colab Lessons

Colab can disconnect during long studies; the package is intended to resume studies through Optuna storage.

Drive mount authorization prompts cannot be fully hidden reliably from notebook code; Colab requires user consent for Drive access.

The notebook currently uses local checkpoint directories to avoid filling Drive with transient trial checkpoints.

DB and log are backed up through `Storage.backup`; checkpoints need separate handling.

If a checkpoint directory is tiny, directly copy the whole directory to Drive; zip compression is usually not worth it for PyTorch `.pt` checkpoints.

## SolarSystemLander Context

The active SSL notebook has used `OBSERVATION_MODE = "8d"` and study series name `solar_system_lander_8d_elise`.

SSL training worlds are `moon`, `mercury`, `mars`, `earth`, and `venus`.

The objective score for SSL is the mean greedy evaluation score across worlds.

A score over 200 across all worlds is possible; a 211 evaluation result was observed in robustness evaluation during this session but was not saved as a model checkpoint.

Early HPO was narrow and focused on `eps_end` and `eps_decay`, yet already found strong candidates; this suggests the setup has real headroom.

The user is optimistic that adaptive training extension can find an "Ue220" model.

## API And Behavior Notes

`StudyRunner.run(...)` should remain easy to call from notebooks; avoid adding required parameters for reporting conveniences.

`Dashboard()` defaults should be usable; currently `Dashboard(render_mode="safe")` is explicit in the SSL notebook, and `training_update_interval_seconds=5.0` does not need to be passed.

`ObjectiveConfig.hooks` is the right place to connect checkpointing and live training progress to the objective.

The SSL notebook must use `ObjectiveHookFactory` in `ObjectiveConfig`; without it, the live training plot has no plotter and stays empty.

Checkpoint window `window` is used both for trailing checkpoint mean and the training plot's trailing mean.

Incumbent score is passed through StudyRunner into checkpoint min score so initial checkpoint threshold starts at the baseline.

`Incumbent.score` is also relevant to checkpointing, not merely dashboard display.

## Testing Notes

Run HPO tests from repo root with:

```powershell
.\dqn\.venv\Scripts\python.exe -m pytest hpo\tests
```

Run DQN tests from repo root with:

```powershell
.\dqn\.venv\Scripts\python.exe -m pytest dqn\tests
```

The old single-environment CartPole timing test can be flaky on this machine; a recent run failed only because it took about 26 seconds while the test expected less than 25 seconds.

For changes touching adaptive training extension, at minimum run:

```powershell
.\dqn\.venv\Scripts\python.exe -m pytest dqn\tests\test_vector_training.py hpo\tests
```

For dashboard changes, run:

```powershell
.\dqn\.venv\Scripts\python.exe -m pytest hpo\tests\evaluation\test_dashboard.py
```

## Open Next Steps

Verify evaluation-best checkpointing in Colab on the next strong run and decide how selected evaluation-best checkpoints should be copied to Drive.

Reconstruct Study Series dashboard context from the existing series DB after Colab reconnect.

Consider a `StudyRunner.show(...)` method later for displaying finished studies or whole series without re-running them.

Consider a smoother dashboard rendering mode only after the safe throttled mode is no longer good enough.

Consider whether robust selection should remain mean-based or add a top-potential mode such as max or top-k mean.

Consider copying only incumbent/evaluation-best checkpoints to Drive automatically at study boundaries.

Keep the package understandable: if a feature starts feeling clever, write the smallest useful version first.
