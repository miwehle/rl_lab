# NN Video Overlay

Goal: add a Micro-Elise network visualization as a bottom overlay band to SolarSystemLander videos, while keeping notebooks thin and implementation logic in `nn_viz`.

## Context

The existing HPO video notebook already records useful lander videos. `nn_viz` can already compute an activity-based `NetworkLayout` and render it as a static PNG. The first video step should combine these two existing capabilities: record a normal lander video, but render the static NN layout into the lower part of each frame.

## Placement

The implementation belongs in `nn_viz/src/nn_viz/video.py`. The notebook belongs in `nn_viz/notebooks`, because this is an NN visualization workflow rather than a distillation training workflow. This design note lives under `distillation/designs` because the current subject is Micro-Elise/Teacher comparison from the distillation work.

## First Slice

Add a small public API:

```python
record_network_overlay_video(
    q_net,
    env_factory,
    layout,
    *,
    world: str,
    seed: int,
    output_path,
    max_steps: int = 1000,
    overlay_height_ratio: float = 0.32,
    overlay_alpha: float = 0.70,
)
```

The first implementation renders the NN layout once, caches it as an RGBA image, alpha-blends it into the bottom band of every rendered lander frame, and lets the normal video path write the resulting frames.

## Rendering Model

The overlay should behave like the existing wind/vector overlays conceptually: it becomes part of the rendered frame before the frame is written to video. The NN band is placed at the bottom, full width, so it uses the screen's wide aspect ratio and normally does not cover the lander.

The static overlay uses the existing `compute_activity_layout(...)` result and the existing plotting style. This avoids a second layout model. Later live activation rendering should reuse the same node positions and edge selection.

## Notebook

Create a thin notebook, likely `nn_viz/notebooks/micro_elise_nn_video.ipynb`, with only:

```text
setup
load-micro-elise
build-layout
record-video
```

The notebook should not contain video composition logic.

## Stages

1. Static overlay: render the existing NN layout once and blend it into the bottom band of the lander video.

2. Smooth live overlay: render the finished NN visualization as an overlay in the moving video, showing the network state dynamically during the landing. The moving video should use smoothed activations, for example a rolling mean over the last `0.5 s`, because raw per-step activations would likely flicker more than they inform. Node brightness comes from smoothed activations, output highlighting from smoothed Q-values/current action, and edge alpha from smoothed current source activation.

3. Freeze/storyboard mode: optionally pause the video at selected steps or create a jump-cut sequence of held still frames. In these still moments, render exact per-step activations, Q-values, action, and edge activity without smoothing. This gives precise decision snapshots while keeping the moving video readable.

If per-frame rendering is too slow, cache or refresh the NN overlay every `n` environment steps as a fallback. The moving overlay should still be based on the current smoothing window; exact raw activations belong in freeze/storyboard frames.

## Non-Goals

Do not refactor the HPO video stack into a generic hook system for this first slice. Do not build a separate video encoder abstraction unless the existing video path cannot accept composed frames cleanly. Do not implement live activations in the same change as the first static overlay.
