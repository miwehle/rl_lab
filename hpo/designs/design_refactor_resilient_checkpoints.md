# Refactor-Resilient Checkpoints

## Goal

HPO best-eval checkpoints should keep working after small package refactorings.

## Problem

`torch.load(..., weights_only=False)` can unpickle Python objects and their old module paths. If checkpoint metadata contains custom objects from moved modules, loading can fail even though the model weights are still usable.

## Format

Use one source of truth for each concern:

```text
best_eval_checkpoint.pt    model state_dict only
best_eval_checkpoint.json  metadata and config
```

The JSON sidecar contains values needed by notebooks and video recording, including `hidden_size`, score, training config, world scores, and study/trial info.

## Implementation

- Save checkpoints as `q_net.state_dict()`.
- Load checkpoints with `torch.load(..., weights_only=True)`.
- Read checkpoint metadata from the JSON file beside the checkpoint.
- Do not read metadata from `.pt` files.
- Do not add runtime compatibility code for old module paths or old checkpoint formats.

## Existing Artifacts

Existing JSON-safe `.pt` files in the old `{version, model_state_dict, metadata}` shape can still be read with `weights_only=True`; use only their `model_state_dict` and ignore `.pt` metadata.

Some older `.pt` files contain pickled custom objects in metadata and cannot be read with `weights_only=True`. Do not add source-code compatibility for them; convert them once outside the normal runtime path.

The active Drive folder `best_checkpoints/solar_system_lander_10d_elise_stp` now contains the converted checkpoint used by the video notebook. Old not-directly-readable checkpoint folders were moved to `G:\Meine Ablage\rl_lab\hpo\best_checkpoints\_archive\old_checkpoints_in_old_format`.

When converting an old checkpoint, first verify the new-format copy and then delete the old artifact only when that is intentional.
