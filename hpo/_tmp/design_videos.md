# Checkpoint Videos

Goal: create short Gymnasium videos for a saved checkpoint, so we can inspect the pilot's flight behavior directly.

KISS API:

```python
video_path = record_checkpoint_video(
    checkpoint_path=...,
    environment_factory=ENV_FACTORY,
    world="venus",
    seed=10_000,
    output_dir=COLAB.drive_study_dir / "videos",
    device=device,
)
```

Optional convenience wrapper:

```python
video_paths = record_checkpoint_videos(
    checkpoint_path=...,
    environment_factory=ENV_FACTORY,
    worlds=["earth", "venus"],
    seeds=[10_000, 10_001, 10_002],
    output_dir=COLAB.drive_study_dir / "videos",
    device=device,
)
```

`record_checkpoint_videos` is only a small loop over `record_checkpoint_video`; it does not make rendering faster.

Minimal data flow:

```text
checkpoint.pt + environment_factory + world + seed
-> create single-world env with render_mode="rgb_array"
-> load q_net from checkpoint metadata
-> run one greedy episode
-> Gymnasium RecordVideo writes mp4
```

This is not a historical training-flight reconstruction. It is a reproducible flight test for a concrete saved pilot. That is enough for the current debugging need: inspect whether a checkpoint hovers, crashes, drifts, lands too hard, or handles wind badly.
