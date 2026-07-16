# Refactor-Resilient Checkpoints

## Goal

HPO checkpoints should survive package refactorings. Improving module structure must not make hard-won model artifacts unusable.

## Problem

PyTorch checkpoints can contain pickle data. Pickle may store Python module paths for custom objects, dataclasses, enums, or configs. If a package later moves, for example from `hpo.solar_system_lander` to `hpo.environments.solar_system_lander`, loading an old checkpoint can fail even though the model weights are still valuable.

## Design Rule

Long-lived checkpoints must store only refactor-resilient data:

- model `state_dict`
- primitive metadata: `str`, `int`, `float`, `bool`, `None`, `list`, `dict`
- tensors

Do not store Python objects such as environments, dataclass instances, enums, callables, or custom config objects in durable checkpoint metadata.

## Shape

Preferred archive shape:

```text
best_eval_checkpoint.pt      model weights/state_dict
best_eval_checkpoint.json    score, hidden_size, training config, world scores, study/trial info
```

The JSON sidecar should be the normal source for metadata needed by notebooks and video recording. Loading a video should not need to unpickle old metadata just to discover `hidden_size`.

## Implementation Direction

Keep `save_checkpoint(...)` simple, but make metadata JSON-safe before writing durable artifacts. A small guard such as `json.dumps(metadata)` is enough to catch accidental custom objects early.

If a high-value legacy checkpoint contains pickle references to old modules, do not mutate it. Preserve the original and create a migrated copy beside it using the current durable format.

## Non-Goals

- No runtime compatibility layer for old module paths by default.
- No generic artifact migration framework.
- No deleting, overwriting, renaming, or replacing hard-won best-eval checkpoints.

## KISS Rule

Greenfield code is welcome. Greenfield treatment of valuable artifacts is not. Keep the checkpoint format boring, primitive, and explicit.
