# Detailed Eagle Skin In Videos

Goal: integrate the detailed Eagle lander graphic into videos without bloating the general rendering and video modules.

## Decision

Use a small optional skin hook in `LanderRenderWrapper`.

`LanderRenderWrapper` remains responsible for the Gym-compatible scene: sky, terrain, flags, particles, score state, overlays, and render mode handling.

The detailed Eagle graphic lives in a separate module, for example `hpo.evaluation.lander_skins.eagle`, and is passed into the wrapper as `skin=DetailedEagleSkin()`.

## Why

The detailed Eagle is presentation logic, not environment logic.

Keeping it as a skin keeps `hpo.evaluation.lander_rendering` focused on rendering orchestration and keeps `hpo.evaluation.video` focused on recording videos.

The detailed graphic also carries calibration constants, body/leg anchors, leg rest angles, and nozzle locations. Those are Eagle-specific and should not spread through the generic evaluation modules.

## Rendering Model

Physics stays unchanged.

The Eagle body follows `env.lander`.

The left and right Eagle legs follow the corresponding Gym leg bodies, so Gym suspension and leg breakage remain visible.

The healthy reset pose should reconstruct the original detailed Eagle as closely as possible. When suspension, contact, or breakage happens, the legs may and should move relative to the body.

The detailed Eagle main impulse point is estimated inside the nozzle body, between the visual nozzle exit and the Gym main impulse point. The current calibration difference is small enough to accept for now. The flame should originate from the visual Eagle nozzle; debug/calibration views may still show both impulse-point estimates.

## Proposed API

Notebook use:

```python
from hpo.evaluation.lander_skins.eagle import DetailedEagleSkin

video_paths = record_checkpoint_videos(
    checkpoint_path=CHECKPOINT_PATH,
    environment_factory=ENV_FACTORY,
    worlds=WORLDS,
    seeds=SEEDS,
    output_dir=VIDEO_DIR,
    device=device,
    colors_by_world=world_colors(WORLDS),
    render_overlay=LanderOverlay(),
    render_skin=DetailedEagleSkin(),
)
```

Package flow:

```text
record_checkpoint_videos(..., render_skin=...)
  -> record_checkpoint_video(..., render_skin=...)
  -> LanderRenderWrapper(..., skin=render_skin)
  -> _render_lunar_lander(...)
  -> skin.draw(surface, env)
```

## Module Shape

```text
hpo/src/hpo/evaluation/
    lander_rendering.py
    video.py
    lander_skins/
        __init__.py
        eagle.py
```

`eagle.py` owns the detailed ops imports, anchor constants, leg rest angles, leg side mapping, and draw logic.

`video.py` should only pass `render_skin` through.

`lander_rendering.py` should only know that a skin has a `draw(surface, env)` method.

## KISS Constraints

Do not replace or modify Gym physics.

Do not parse SVG at runtime.

Do not use the old simplified Eagle path for this integration.

Do not build a general sprite engine.

Do not add many visual knobs before the detailed Eagle skin is useful in real audit/showcase videos.

Keep the default renderer unchanged unless a skin is explicitly passed.
