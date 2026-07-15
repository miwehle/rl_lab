# Convention Over Configuration

## Goal

Use the simplest solution that keeps incidental details under the hood and exposes the right levers.

For the video notebook, Colab, Drive, checkpoint filenames, metadata filenames, and video directories are incidental details. The notebook should not show them in the normal path.

## Pattern

Functions that need infrastructure defaults should accept a small cfg object with a useful default:

```python
def record_video(..., cfg=DefaultVideoInfraCfg()):
    ...
```

The default cfg is the convention. It is fully usable by itself and contains the standard infrastructure choices. Users can pass another cfg when they want to leave the convention.

## Video Notebook

These details should disappear from the notebook:

```python
CHECKPOINT_DIR = COLAB.drive_study_dir / "best_checkpoints" / STUDY_NAME
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_eval_checkpoint.pt"
CHECKPOINT_METADATA_PATH = CHECKPOINT_DIR / "best_eval_checkpoint.json"
VIDEO_DIR = COLAB.drive_study_dir / "videos" / STUDY_NAME
```

They belong under the hood, for example in `DefaultVideoInfraCfg`.

`DefaultVideoInfraCfg` should contain infrastructure conventions, not the current study identity. The current `study_name` is a fachlicher Hebel and stays explicit in the notebook or other using code.

## Responsibilities

Keep fachliche cfg visible in the notebook: model, env, world, seed, skin, overlay, and other choices that define what video is wanted.

Hide infrastructure cfg behind the convention: Drive layout, checkpoint directory names, checkpoint filenames, metadata filenames, video directory names, and path joins.

## Shape

The normal notebook call should move toward:

```python
record_video(DQN, env, study_name=STUDY_NAME, seed=seed)
```

Advanced users may override the infrastructure convention:

```python
record_video(DQN, env, study_name=STUDY_NAME, seed=seed, cfg=custom_video_infra_cfg)
```

KISS rule: do not add a global config framework, singleton, or large generic convention system. Start with one concrete `DefaultVideoInfraCfg` for this video use case.

## Generalization

Apply the same convention-over-configuration shape symmetrically where it fits: training, video recording, and audit workflows should not each invent their own visible path constants.

A small base cfg is welcome when it removes real duplication and names the shared infrastructure convention:

```python
@dataclass(frozen=True)
class InfraCfg:
    drive_study_dir: Path = Path("/content/drive/MyDrive/rl_lab/hpo")
    local_study_dir: Path = Path("/content/rl_lab/hpo/runs")
    best_checkpoints_dir: str = "best_checkpoints"
    videos_dir: str = "videos"
```

Concrete cfg classes can inherit from it when the relationship is natural. Use the `*InfraCfg` naming scheme because these classes are not complete fachliche configs; they hold infrastructure conventions such as runtime setup, paths, filenames, storage layout, backup/restore locations, and artifact directories.

```python
@dataclass(frozen=True)
class TrainInfraCfg(InfraCfg):
    database_suffix: str = ".db"
    log_suffix: str = ".log"

    def database_path(self, study_name: str) -> Path: ...
    def drive_database_path(self, study_name: str) -> Path: ...
    def log_path(self, study_name: str) -> Path: ...
    def drive_log_path(self, study_name: str) -> Path: ...


@dataclass(frozen=True)
class VideoInfraCfg(InfraCfg):
    checkpoint_name: str = "best_eval_checkpoint.pt"
    checkpoint_metadata_name: str = "best_eval_checkpoint.json"

    def checkpoint_dir(self, study_name: str) -> Path: ...
    def checkpoint_path(self, study_name: str) -> Path: ...
    def checkpoint_metadata_path(self, study_name: str) -> Path: ...
    def video_dir(self, study_name: str, purpose: str | None = None) -> Path: ...


@dataclass(frozen=True)
class AuditInfraCfg(VideoInfraCfg):
    audit_video_scope: str = "failure_audit"

    def audit_video_dir(self, study_name: str) -> Path: ...
```

`TrainInfraCfg` should cover training infrastructure defaults such as local and Drive database/log paths, restore, backup, and file naming conventions.

`VideoInfraCfg` should cover video infrastructure defaults such as best-eval checkpoint paths, checkpoint metadata paths, video output directories, and video filename conventions.

`AuditInfraCfg` should cover audit infrastructure defaults such as audit video scope, audit output directories, and reuse of checkpoint/video conventions needed by failure-audit workflows.

Keep fachliche inputs explicit in the using code: `study_name`, model, env, world, seed, skin, overlay, training hyperparameters, and audit selection policy are not hidden in the under-the-hood cfg by default.

The name `TrainInfraCfg` avoids suggesting a full training configuration: learning rate, episodes, model architecture, early stopping, and similar training decisions stay outside. `VideoInfraCfg` avoids suggesting render semantics: skin, overlay, world, and seed stay outside. `AuditInfraCfg` avoids suggesting audit policy: what to inspect or rank stays outside; only audit infrastructure placement belongs there.

KISS rule for this generalization: use inheritance only for real shared infrastructure convention. Do not build a large generic config framework, registry, singleton, or speculative hierarchy.

## Future Domain Configs

The `*InfraCfg` classes deliberately cover only infrastructure details. This naming keeps room for future fachliche configs without mixing concerns.

A possible future naming scheme is:

```python
TrainingSpec   # fachliche training configuration: what to train
VideoSpec      # fachliche video configuration: what video to record
AuditSpec      # fachliche audit configuration: what to inspect or rank
```

`*Spec` is short and emphasizes the domain intent. It should hold domain decisions such as training hyperparameters, model choices, worlds, seeds, rendering choices, or audit selection policy when it becomes useful to bundle them.

`*DomCfg` is another possible naming scheme when the code should say "domain config" more explicitly:

```python
TrainingDomCfg
VideoDomCfg
AuditDomCfg
```

For now this is only future music. Do not add fachliche config classes until they reduce current complexity or make a real API easier to use.

## Clarification: No Visible Infrastructure Config in Notebook Normal Path

The notebook normal path should show no infrastructure config object and no infrastructure setup object.

Do not replace visible path constants with visible objects such as `COLAB`, `IO`, `TECH`, `VideoInfraCfg`, or `HpoInfraCfg` in notebook code. That is still visible infrastructure configuration.

Infrastructure details belong under the hood in default `*InfraCfg` conventions passed through optional `cfg=` parameters on notebook-facing functions.

Design the public API from the desired notebook cell outward. Existing function names, signatures, and helper boundaries are not constraints.

Preferred normal shape:

```python
record_video(DQN, env, study_name=STUDY_NAME, seed=seed)
```

Advanced override shape:

```python
record_video(DQN, env, study_name=STUDY_NAME, seed=seed, cfg=custom_video_infra_cfg)
```

No Colab, Drive, path, logging, device, backup, checkpoint filename, or artifact layout code should appear in the notebook normal path.

Keep fachliche DL/HPO choices explicit: model, env, study name, worlds, seeds, rendering intent, HP ranges, training decisions, and audit policy.

KISS rule: optimize for the simplest notebook end state with the best learning/HPO signal. Hide infrastructure mechanics by default; expose them only as explicit advanced overrides.
