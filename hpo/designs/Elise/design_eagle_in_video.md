# Eagle In Videos

Goal: keep video generation fast and faithful by default, but allow nicer showcase videos with an Eagle-inspired lander.

## Decision

The default video rendering stays the spartan Gymnasium lander.

Video recording may opt into a lander style:

```python
lander_style="gymnasium" | "eagle" | "simplified_eagle"
```

`"gymnasium"` uses the current fixture-based drawing.

`"eagle"` uses the detailed Eagle drawing from `hpo/_experimental/lander_rendering/detailed/eagle_lander_pygame.py`.

`"simplified_eagle"` uses the compact hand-drawn Pygame version from `hpo/_experimental/lander_rendering/simple/simplified_eagle_pygame.py`.

## Why

The detailed Eagle looks much closer to the original lunar module and is good for showcase videos.

It costs roughly `2 ms/frame` in a local Pygame microbenchmark, about `2 s` for a `1000` frame video. That is acceptable for selected videos but not ideal as the always-on default.

The simplified Eagle is much cheaper, roughly `0.2 ms/frame`, and may be useful when the detailed Eagle is too slow or visually too busy.

## Rendering Model

Physics stays unchanged.

Only the visual representation of the lander changes.

The Gymnasium terrain, particles, flags, overlay text, score, and world colors stay as they are.

For `"eagle"` and `"simplified_eagle"`, the renderer should draw the chosen Eagle surface at the current Gymnasium lander pose.

The first implementation should prefer a simple, visually useful placement over perfect physical fidelity.

## API Sketch

Notebook use:

```python
video_paths = record_checkpoint_videos(
    checkpoint_path=CHECKPOINT_PATH,
    environment_factory=ENV_FACTORY,
    worlds=WORLDS,
    seeds=SEEDS,
    output_dir=VIDEO_DIR,
    device=device,
    colors_by_world=world_colors(WORLDS),
    render_overlay=LanderOverlay(),
    lander_style="eagle",
)
```

Package flow:

```text
record_checkpoint_videos(...)
  -> record_checkpoint_video(..., lander_style=...)
  -> LanderRenderWrapper(..., lander_style=...)
  -> _render_lunar_lander(...)
```

## KISS Constraints

Do not replace the environment physics.

Do not parse SVG at runtime.

Do not make the detailed Eagle the default.

Do not add many style knobs before the three styles above have proven useful.

The current target is better videos, not a general sprite engine.
