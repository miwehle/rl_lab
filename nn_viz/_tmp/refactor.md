# NN Viz Video Refactor

## Ziel

`nn_viz.video` ist mit ca. 840 LOC zu gross. Die sinnvolle Sollbruchstelle sind die zwei Haupt-Clients:

- Video-Recording mit NN-Overlay.
- Offline PNGs aus gespeicherten Trace-Dateien.

Die Aufteilung soll Code duplizieren vermeiden, aber kein grosses Utility-Sammelbecken erzeugen.

## Zielstruktur

```text
nn_viz/video.py
  record_video(...)

nn_viz/trace.py
  render_trace_step(...)
  render_trace_diff(...)

nn_viz/_rendering.py
  gemeinsame Render-Mechanik
```

## video.py

Bleibt fuer Recording-spezifisches verantwortlich:

- `record_video(...)`
- `_VideoTrace`, wenn es nur beim Recording geschrieben wird.
- `_StateAverager`
- `_initial_state(...)`
- `_NetworkOverlayWrapper`
- `_compose_bottom_overlay(...)`
- `_draw_step_label(...)`
- `_hold_final_frame(...)`

Alles hier soll mit laufender Env, Video-Frames, Step-Label oder Trace-Schreiben waehrend des Recordings zu tun haben.

## trace.py

Enthaelt offline Trace-PNG-Logik:

- `render_trace_step(...)`
- `render_trace_diff(...)`
- `_load_trace_state(...)`
- `_trace_state_from_arrays(...)`
- `_trace_scales_from_arrays(...)`
- `_diff_state(...)`
- `_render_diff_layout_rgba(...)`, falls die Diff-Farblogik nur hier gebraucht wird.

Alles hier soll aus gespeicherten Trace-Dateien arbeiten und keine Video-Recording-Abhaengigkeiten haben.

## _rendering.py

Enthaelt nur gemeinsam genutzte Render-Mechanik, nicht die fachliche Video- oder Trace-PNG-Orchestrierung.

Gute Kandidaten:

- `_NetworkState`
- `_EdgeStyle`
- `_render_layout_rgba(layout, node_fill, node_outline, edge_style, ...)`
- Zeichenfunktionen fuer Kanten, Nodes und Labels.
- Font-/aggdraw-/RGBA-Helfer, soweit sie von beiden Pfaden gebraucht werden.
- kleine Wert-/Scale-Helfer wie `_node_value(...)`, `_source_value(...)`, `_scale_value(...)`, `_input_scale(...)`, falls beide Pfade sie nutzen.

Wichtig: `_rendering.py` soll nicht einfach `_render_state_layout_rgba(...)` und `_render_diff_layout_rgba(...)` als fertige Fachfunktionen sammeln, wenn eine davon nur einen Client hat. Besser ist eine gemeinsame Primitive:

```python
_render_layout_rgba(
    layout,
    node_fill=...,
    node_outline=...,
    edge_style=...,
    edge_renderer=...,
    label_mode=...,
)
```

Dann definieren `video.py` und `trace.py` jeweils ihre eigene Bedeutung von Node-/Edge-Styles und teilen nur die Zeichenmaschine.

## Vermeidung von dupliziertem Code

Das Ziel ist nicht, zwei fast gleiche Renderer zu bauen:

```text
video.py
  zeichnet Canvas, Kanten, Nodes, Labels

trace.py
  zeichnet Canvas, Kanten, Nodes, Labels nochmal fast gleich
```

Das waere duplizierter Zeichen-Code und wuerde spaetere Aenderungen teuer machen.

Stattdessen soll `_rendering.py` genau den gemeinsamen mechanischen Teil kapseln:

```text
_rendering._render_layout_rgba(...)
  legt RGBA-Canvas an
  transformiert Layout-Koordinaten in Pixel
  zeichnet Kanten
  zeichnet Nodes
  zeichnet Labels
  gibt RGBA-Array zurueck
```

Die fachliche Bedeutung der Farben bleibt aber ausserhalb:

```text
video.py
  baut node_fill(node) fuer normalen NN-State
  baut edge_style(edge) fuer normalen NN-State
  ruft _rendering._render_layout_rgba(...)

trace.py
  baut node_fill(node) fuer Trace-Step oder Diff
  baut edge_style(edge) fuer Trace-Step oder Diff
  ruft _rendering._render_layout_rgba(...)
```

Damit teilen beide Clients die Zeichenmaschine, aber nicht ihre fachliche Style-Logik. Das ist der eigentliche Punkt der Aufteilung: keine Duplikation der Low-Level-Zeichnung, aber auch kein Verschieben von trace-spezifischer Diff-Logik in ein gemeinsames Modul.

## Public API

Die Notebook-API bleibt ueber `nn_viz.__init__`:

- `record_video`
- `render_trace_step`
- `render_trace_diff`

Nach dem Split importiert `nn_viz.__init__` diese aus den neuen Modulen. `nn_viz.video` muss nicht mehr die Trace-PNG-Funktionen enthalten.

## KISS-Leitplanke

Nur verschieben, was eine klare gemeinsame Nutzung hat. Wenn eine Funktion nur von `video.py` oder nur von `trace.py` verwendet wird, bleibt sie als private Funktion dort.
