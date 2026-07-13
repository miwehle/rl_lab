# Lander Rendering Split

## Goal

Split the current lander rendering code by what is visible in the video: scene, lander, windsocks, vectors, and text. Keep the design simple: small modules, plain functions, no compatibility shims, and no framework layer.

## Target Tree

```text
hpo/src/hpo/evaluation/rendering/
+-- lander/
    +-- __init__.py
    +-- scene.py
    +-- colors.py
    +-- env_state.py
    +-- lander.py
    +-- windsocks.py
    +-- overlay/
        +-- __init__.py
        +-- overlay.py
        +-- text.py
        +-- vectors.py
    +-- skins/
        +-- __init__.py
        +-- eagle.py
        +-- _eagle_body_pygame.py
        +-- _eagle_legs_pygame.py
```

## Module Roles

`scene.py`: Owns `LanderRenderWrapper`, frame creation, render order, Gym terrain drawing, flag positions, and final RGB frame conversion.

`colors.py`: Owns `LanderColors`, world color lookup, and color constants.

`env_state.py`: Owns `EnvState`, the render-facing view of the Gym/env state. It reads Gym/env internals once per frame and prepares simple values for the drawing modules: world, gravity, score, wind, turbulence, kick, step count, mass, inertia, body positions, angles, contacts, and screen positions. Details belong in the module and class docstrings.

`lander.py`: Owns the visible lander drawing decision: default Gym lander polygons or optional skin. It hides the default Gym lander when a skin is active.

`overlay/`: Owns all non-world overlay drawing: text lines, wind/kick/turbulence vectors, and overlay composition.

`overlay/overlay.py`: Orchestrates visible overlay elements. It should stay thin: read `EnvState`, draw text lines, draw vectors, and call specialized modules.

`overlay/text.py`: Draws text and labels with shadow. It has no Gym/env dependencies.

`overlay/vectors.py`: Draws wind, kick, and turbulence vector/tacho visualizations from simple values. It owns the small vector math needed for drawing.

`windsocks.py`: Draws windsocks at flag poles from wind values. It owns the wind-strength scaling needed for the windsock shape.

`skins/`: Owns lander-specific visual skins. The current detailed Eagle skin and generated Eagle ops live here.

## Module Tree

```text
scene.py
+-- colors.py
+-- env_state.py
+-- lander.py
|   +-- skins/
|       +-- eagle.py
|       +-- _eagle_body_pygame.py
|       +-- _eagle_legs_pygame.py
+-- windsocks.py
+-- overlay/
    +-- overlay.py
    +-- text.py
    +-- vectors.py
```

This is the intended ownership tree between modules. Each module appears exactly once. There is no cross-cutting helper module in the first split; small calculations stay close to the visual element or env-state extraction that needs them.

## Rules

- No upward imports: low-level modules (`overlay/text.py`, `colors.py`) must not import high-level modules (`scene.py`, `overlay/overlay.py`, `lander.py`).
- `env_state.py` is the main place that reads Gym/env internals for rendering data.
- `vectors.py` and `windsocks.py` should draw from simple values, not inspect raw envs unless the value is local to that visual element.
- `scene.py` owns render order.
- `overlay/overlay.py` owns overlay composition, not low-level drawing details.
- `lander.py` owns lander visibility and skin selection, not terrain, text, or vector overlays.
- Keep public API small and new. No backward-compatibility shim for the old `hpo.evaluation.lander_rendering` module.

## Refactor Plan

1. Move current `lander_skins` into `rendering/lander/skins`.
2. Create `rendering/lander` modules with minimal exports from `__init__.py`.
3. Move code mechanically by visible role, keeping behavior unchanged.
4. Update imports in production code, notebooks, and tests to the new package.
5. Run `hpo\tests`.
6. Only after tests are green, consider further simplification inside modules.
