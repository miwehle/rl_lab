# Vereinfachung von: video.py, Notebooks, hpo/evaluation/rendering

## Vorgehen


1. **Neue einfache Video-Frontdoor setzen**
   - Ziel jetzt:
     ```python
     record_video(ckpt, model, env, render_cfg=None)
     ```
   - `env` ist fertig gebaut.
   - `model` bleibt vorerst explizit.
   - `render_cfg` ist optional.

2. **Tests für diese Frontdoor schreiben/umbauen, damit der Kern sitzt**

3. **Notebook und `video.py` auf diese einfache API ausrichten**
   - Notebooks sollen die einfache Story zeigen.
   - `video.py` soll möglichst wenig Rendering-Details direkt kennen.
   - Bestehende kompliziertere Funktionen nur so lange behalten, wie sie beim Übergang helfen.

4. **Altes kompliziertes Video-Zeug entfernen**
   - Keine Rückwärtskompatibilität.
   - Kein Paralleluniversum von alter und neuer API.
   - Sobald Notebook/tests umgestellt sind: weg damit.

5. **Rendering-API-Grenze explizit machen**
   - Public nur, was außerhalb von Rendering wirklich genutzt wird.
   - Reexport in `__init__.py` nur für diese Namen.
   - Alles andere internal machen: `_module`, `_subpackage`, `_function`, `_Class` je nach Ebene.
   - Stand jetzt, YAGNI. Keine hypothetischen Erweiterungspunkte.
   - Erst nach der Video-/Notebook-Vereinfachung final ziehen, weil dann klar ist, was wirklich public bleiben muss.

## Rendering-Ziel-API

KISS-Kandidat:

```
cfg = render_config(worlds, overlay=True, skin="detailed_eagle")
record_video(ckpt, model, env, render_cfg=cfg)
```

## Kriterien

1. **API-Einfachheit**
   - Notebook-Hauptpfad sollte auf etwa diese Form schrumpfen:
     ```python
     env = ...
     cfg = render_config(...)
     record_video(ckpt, model, env, render_cfg=cfg)
     ```
   - Keine direkten Imports von Wrappern/Colors/Skins im Notebook.

2. **Kopplung**
   - `video.py` importiert aus Rendering maximal `RenderConfig`/Protocol oder gar nichts Konkretes.
   - Ziel: keine SSL-spezifischen Namen in `video.py`.

3. **LOC / Surface**
   - `video.py` netto kürzer oder zumindest deutlich weniger verzweigt.
   - `solar_system_lander/__init__.py` reexportiert nur Namen, die außerhalb wirklich gebraucht werden.

Vorteile ggü Ist-Zustand: weniger Imports, kleinere Public Surface, einfacherer Notebook-Code.