# Lander Rendering

## Ziel

Checkpoint-Videos sollen optional andere Renderfarben fuer Himmel, Boden, Lander und Flagge verwenden koennen, damit SolarSystemLander-Videos pro Welt intuitiver lesbar werden.

Die Aenderung betrifft nur die Darstellung aufgezeichneter Videos. Training, Evaluation, Physik, Observationen, Rewards und Checkpoints bleiben unveraendert.

## API

`record_checkpoint_video(...)` bekommt einen optionalen Parameter:

```python
render_colors: LanderColors | None = None
render_overlay: LanderOverlay | None = None
```

`record_checkpoint_videos(...)` bekommt einen optionalen Parameter:

```python
colors_by_world: Iterable[LanderColors | None] | None = None
render_overlay: LanderOverlay | None = None
```

Wenn `colors_by_world` gesetzt ist, gehoert `colors_by_world[i]` zu `worlds[i]`. Die Laenge muss der Laenge von `worlds` entsprechen; sonst wird ein `ValueError` geworfen.

Wenn keine Farben uebergeben werden, bleibt das bestehende Gymnasium-Rendering exakt der Default-Pfad.

## Farben

Die Farben werden in einem eigenen Modul definiert:

```text
hpo/src/hpo/evaluation/lander_rendering.py
```

```python
RGB = tuple[int, int, int]

@dataclass(frozen=True)
class LanderColors:
    sky: RGB = (0, 0, 0)
    ground: RGB = (255, 255, 255)
    ground_outline: RGB = (0, 0, 0)
    lander_fill: RGB = (128, 102, 230)
    lander_outline: RGB = (77, 77, 128)
    flag_pole: RGB = (255, 255, 255)
    flag: RGB = (204, 204, 0)

@dataclass(frozen=True)
class LanderOverlay:
    text_color: RGB = (255, 255, 255)
    shadow_color: RGB = (0, 0, 0)
```

Die Defaults sind die aktuellen Gymnasium-Farben. Konkrete Weltpaletten wie Earth, Venus, Mars, Moon oder Mercury werden nicht in `video.py` definiert, sondern vom Caller uebergeben, zum Beispiel im Notebook.

Wenn `render_overlay` gesetzt ist, zeigt das Video links oben die verfuegbaren statischen Werte: Weltname, Betrag der Gravitation mit einer Nachkommastelle und `m/s²`, `wind` als maximale seitliche Beschleunigung mit `m/s²`, `turb` als maximale Winkelbeschleunigung mit `rad/s²`, und `kick` als geschaetztes Delta-v aus dem initialen Reset-Kraftstoss mit `m/s`.

## Umsetzung

`lander_rendering.py` enthaelt einen `LanderRenderWrapper`, der nur `render()` ersetzt und sonst an das gewrappte Environment delegiert.

Der Wrapper macht kein Frame-Postprocessing. Er verwendet eine kleine lokale Variante der Gymnasium-LunarLander-Renderlogik und ersetzt nur die hardcodierten Farbliterale durch `LanderColors`.

`video.py` bleibt Orchestrierungscode: Checkpoint laden, Environment bauen, bei gesetzten Farben wrappen, dann `RecordVideo` starten.

```python
env = make_env()
if render_colors is not None:
    env = LanderRenderWrapper(env, render_colors)
env = RecordVideo(env, ...)
```

## Tests

Die bestehenden Video-Tests bleiben erhalten.

Neue Tests pruefen, dass `record_checkpoint_video(...)` ein Env bei gesetzten `render_colors` wrappt, dass `record_checkpoint_videos(...)` `colors_by_world` in Welt-Reihenfolge durchreicht, und dass eine falsche `colors_by_world`-Laenge abgelehnt wird.

Ein kleiner Rendering-Test kann mit einem echten `LunarLander-v3` und `rgb_array` pruefen, dass ein gerenderter Frame mit Custom-Farben nicht leer ist und mindestens die erwarteten Himmel- und Bodenfarben enthaelt.
