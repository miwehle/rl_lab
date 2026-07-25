# NN Viz

Neural-network visualization helpers for Elise-like DQN policies.

## Notebook

Useful notebooks:

- `nn_viz/notebooks/micro_elise_nn_video.ipynb` for landing videos with NN state overlays.
- `nn_viz/notebooks/elise_dv_ablation.ipynb` for input-acceleration ablation checks.

## Public API

NN Viz follows the same two-level public API convention as `hpo`.

`api-public` objects are objects intended for notebooks and external clients. They are re-exported from `nn_viz/__init__.py`, so notebook code can treat `nn_viz` as the user-facing API surface.

`module-public` objects are names without a leading `_` in a public module or public class. They may be used by higher-level package code without being re-exported from package `__init__.py` files.

Module names should get a leading `_` only when the whole module is a private implementation detail. Public subpackages such as `nn_viz.layout` may contain public modules such as `activity`, `semantic`, and `types`; they should not be hidden just because notebooks usually import through `nn_viz` or `nn_viz.layout`.

Direct tests should usually target only one of these two public API levels. Names with a leading `_`, and members of a private surrounding structure, are private implementation details and should usually be tested through their public users instead.

## Local Tests

Run from the repository root:

```powershell
dqn\.venv\Scripts\python.exe -m pytest nn_viz\tests
```

