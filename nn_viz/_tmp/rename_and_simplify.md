# NN Viz Naming Cleanup

## Ziel

Die Namen sollen klar zwischen Datenquelle, Zustand und Rendering unterscheiden. `trace` bleibt als bewusste Abweichung vom normalen Video-Recording erhalten. `live` ist in `nn_viz.video` dagegen meistens redundant, weil der Zweck von `record_video(...)` genau das NN-Overlay waehrend des laufenden Fluges ist.

## Begriffe

- `trace`: Zustand wird nachtraeglich aus einer gespeicherten Step-Zeitreihe geladen.
- `state`: neutraler NN-Zustand, der gerendert werden kann, unabhaengig davon, ob er aus `live` oder `trace` kommt.
- `layout`: raeumliche Anordnung des NN.
- `overlay`: RGBA-Bild, das ins Lander-Video komponiert wird.
- `window_steps`: Anzahl der Steps fuer den Rolling Mean im Video oder fuer den rueckwaerts gemittelten Trace-Snapshot.

## Problem

`_LiveOverlayState` wird nicht nur fuer Live-Video genutzt, sondern auch fuer Trace-Step-PNGs und Trace-Diff-PNGs. Der Name ist daher zu eng.

`_render_dynamic_layout_rgba(...)` ist der einzige echte `dynamic`-Symbolname. Die Funktion ist nicht speziell dynamisch; sie rendert ein Layout mit uebergebenen Style-Funktionen.

`live_overlay` ist als Parameter von `record_video(...)` fachlich redundant. Ohne NN-Overlay ist `hpo.evaluation.video.record_video(...)` zustaendig; `nn_viz.video.record_video(...)` sollte immer ein NN-Overlay erzeugen.

## Vorschlag

Kleiner, sinnvoller Refactor fuer Namen und API:

- `compute_live_scales` -> `compute_scales`
- `live_scales` -> `scales`
- `live_window_steps` -> `window_steps`
- `_LIVE_WINDOW_STEPS_DEFAULT` -> `_WINDOW_STEPS_DEFAULT`
- `_LiveOverlayState` -> `_NetworkState`
- `_LiveOverlayAverager` -> `_StateAverager`
- `_render_live_layout_rgba(...)` -> `_render_state_layout_rgba(...)`
- `_render_dynamic_layout_rgba(...)` -> `_render_state_layout_rgba(...)`
- `_initial_live_state` -> `_initial_state`
- `_live_node_color` -> `_node_color`
- `_skip_live_edge` -> `_skip_edge`
- `_draw_live_labels` -> `_draw_labels`
- `_LIVE_LAYOUT_X_PAD` -> `_LAYOUT_X_PAD`
- `_LIVE_LAYOUT_TOP_MARGIN_RATIO` -> `_LAYOUT_TOP_MARGIN_RATIO`
- `_LIVE_LAYOUT_BOTTOM_MARGIN_RATIO` -> `_LAYOUT_BOTTOM_MARGIN_RATIO`

Vereinfachung ueber Namen hinaus:

- `live_overlay` aus `record_video(...)` entfernen.
- Den Branch `if live_overlay else None` entfernen; `record_video(...)` rendert immer ein NN-State-Overlay.
- Statischen Overlay-Fallback aus `video.py` entfernen. Das ist der alte Stage-1-Modus: ein einmal gerendertes Layout-PNG ohne Aktivierungen.
- `_render_layout_rgba(...)` entfernen, wenn es nur noch fuer den statischen Video-Fallback genutzt wird.
- `_crop_to_visible_alpha(...)` entfernen, falls es danach keine andere Nutzung mehr hat.
- `plot_network_layout`-Import aus `video.py` entfernen, wenn der statische Fallback wegfaellt.
- Cache-Logik fuer ein statisches Overlay aus `_NetworkOverlayWrapper` entfernen; der Wrapper bekommt dann immer einen Overlay-Provider.
- `plot.py` bleibt fuer statische Layout-PNGs/Inspektion erhalten.

`trace` bleibt fuer gespeicherte Flug-Zeitreihen:

- `_VideoTrace`
- `_load_trace_state(...)`
- `render_trace_step_png(...)`
- `render_trace_diff_png(...)`
- `_trace_state_from_arrays(...)`
- `_trace_scales_from_arrays(...)`

Damit wird die gemeinsame Pipeline lesbarer:

```text
record_video -> _NetworkState -> _render_state_layout_rgba
trace file   -> _NetworkState -> _render_state_layout_rgba
trace diff   -> _NetworkState -> _render_state_layout_rgba with diff styles
```
