


Ich finde interessant:
## Notebook

cell: setup
`LAYOUT_BUILDER = compute_*_layout`

cell: build-layout
`layout = LAYOUT_BUILDER(...)`

cell: record-video
`record_video(..., layout, ...)`

cell: render-step-png
`render_trace_step_png(..., layout, ...)`

cell: render-diff-png
`render_trace_diff_png(..., layout, ...)`
## src
  
nn_viz
  layout
    activity.py
    semantic.py
  video.py
    `record_video`

`compute_*_layout(...) -> NetworkLayout`

`class NetworkLayout`
- nodes
- edges


Builder

Layout
