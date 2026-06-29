# HPO Insights

## SolarSystemLander: 8D, 9D, 11D

Die wichtigste Erkenntnis zum Observation-Modus ist bisher: Mit 9D und geeigneten HPs geht es auf der Erde. Der Rest bleibt unsicher, weil fruehere 8D- und 11D-Vergleiche vermutlich stark von noch unpassenden HPs und zu kurzem Training gepraegt waren.

Der schwache 11D-Lauf beweist nicht, dass 11D schlechter ist: die HPs und das Trainingsbudget waren vermutlich der groessere Hebel. 9D koennte reichen, vielleicht sogar 8D mit guten HPs.

8D zeigte mehr Potenzial. Ein 211er wurde in einer fruehen Robustness-Evaluation beobachtet, war aber nicht als Modellcheckpoint gesichert. Der beste noch gesicherte Pilot ist der 8D-Elise-bunt-Checkpoint `trial_0009_eval_best.pt` mit Score `180.31` ueber alle fuenf inneren Welten.

Die Epsilon-Region um `eps_end ~= 0.05..0.06` und `eps_decay ~= 40_000..80_000` kann starke 8D-Einzelpiloten erzeugen: eine fokussierte S3-Folgestudie fand dort erneut einen Kandidaten um `175`. Einzelne HP-Saetze bleiben aber stark seed-abhaengig, und 8D bleibt strukturell fragwuerdig, weil dem Netz der Weltkontext fehlt.

Beste gefundene 8D-HPs fuer konkrete Spitzenpiloten:

| Quelle | Score | `eps_end` | `eps_decay` | Hinweis |
|---|---:|---:|---:|---|
| `8d_elise_bunt`, Trial 9 | 180.31 | 0.055604 | 67181 | bestes gesichertes Modell |
| `8d_elise_optevery2`, Trial 18 | 153.54 | 0.038376 | 34218 | bester Trial in dieser DB, robust etwa 105 |

Die aktuelle Arbeitshypothese liegt auf 9D: originale 8D-LunarLander-Observation plus `gravity / 12`, aber ohne Wind- und Turbulenzparameter. Das koennte der Sweet Spot sein: genug Weltkontext fuer das Netz, aber keine fragwuerdigen Zusatzsignale.

Eine interessante Folgestudie waere, 8D, 9D und 11D mit inzwischen guten HPs sauber zu vergleichen. Dann liesse sich besser erkennen, was die Zusatzdimensionen wirklich bringen.

Fuer eine erste 9D-Studie ist ein moderat breiter Suchraum plausibel:

```python
trial.suggest_float(HP.EPS_END, 0.045, 0.065)
trial.suggest_int(HP.EPS_DECAY, 40_000, 80_000, log=True)
```

Der konkrete 180er war ein wichtiger historischer Meilenstein: Er gab die Zuversicht, dass ein brauchbarer Innenplaneten-Pilot moeglich ist. Die aktuell staerkere Erkenntnis ist aber, dass 9D mit geeigneten HPs auf der Erde ueber `200` Gym Score schafft.

## SolarSystemLander: Erde ist lernbar

==Eine Earth-only-Studie zeigte: Das Problem ist nicht, dass LunarLander/DQN die Erde grundsaetzlich nicht schafft. Mit breiterer HPO-Suche fand Optuna schon sehr frueh Trials ueber `200` Gym Score auf der Erde.==

Der wichtige Schritt war, Optuna wieder mehr Freiheitsgrade zu geben statt nur lokal an `eps_end` und `eps_decay` zu drehen. In der erfolgreichen Earth-only-Suche wurden zusaetzlich `learning_rate`, `replay_memory_capacity`, `batch_size` und `num_episodes` optimiert. ==Besonders `num_episodes=1000` scheint wichtig zu sein; `500` Episoden waren fuer die Erde oft zu kurz.==

Gute Earth-HPs lagen bisher ungefaehr hier:

```python
eps_end ~= 0.036
eps_decay ~= 31_000..39_000
learning_rate ~= 4.5e-4..7.5e-4
replay_memory_capacity ~= 35_000..56_000
batch_size = 512
num_episodes = 1000
```

==Zwei konkrete Earth-only-Kandidaten sind besonders wertvoll und sollten als Startpunkte fuer Folgestudien im Blick bleiben:==

| Trial | Score | `eps_end` | `eps_decay` | `learning_rate` | `replay_memory_capacity` | `batch_size` | `num_episodes` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 212.0 | 0.03600291791964673 | 38513 | 0.0004536691702712524 | 35535 | 512 | 1000 |
| 3 | 218.3 | 0.036489618400795065 | 31093 | 0.0007486419514197401 | 56280 | 512 | 1000 |

Interpretation: Erde und Venus sind vermutlich nicht unlandbar, sondern brauchen mehr Training und andere HPs als die kleinen-g-Welten. Die Mehrwelten-HPO darf deshalb Earth/Venus nicht vorschnell als hoffnungslose Faelle behandeln. Manchmal muss man Optuna einfach die Leine laenger lassen.

==Vorlaeufige Evidenz aus der noch laufenden Earth-only-Studie: Gegenueber den frueheren Mehrwelt-Studien waren vermutlich vor allem `num_episodes` und `replay_memory_capacity` schlecht gesetzt.== `500` Episoden waren fuer Erde oft zu kurz; der alte Replay Buffer `400_000` wirkt im Vergleich zu erfolgreichen Earth-Trials um `35_000..56_000` zu gross. Fuer fuenf Welten koennte ein Bereich um etwa `150_000..300_000` plausibler sein als `400_000`.

Danach kommen wahrscheinlich `eps_end` und `eps_decay`: Die guten Earth-Trials liegen bei `eps_end ~= 0.033..0.0365` und `eps_decay ~= 31_000..39_000`, waehrend fruehere Mehrwelt-Laeufe eher bei `eps_end=0.05` und `eps_decay=43_214` starteten. `learning_rate` ist dagegen noch uneindeutig.

==Hypothese mit vorlaeufiger, noch unvollstaendiger Evidenz: Wer auf der Erde einen `200+`-Piloten will, muss ihn mindestens etwa `900` Episoden trainieren.== In den bisherigen Top-Earth-Trials erreichte der Training-Mean ueber 100 Episoden `160+` erst sehr spaet oder gar nicht, obwohl die greedy Evaluation danach teils ueber `200` lag.

Fuer 5-Welten-Training ist entscheidend: `num_episodes` zaehlt Gesamt-Episoden, nicht Episoden pro Welt. `1000` Gesamt-Episoden bedeuten bei gleich verteilter Weltwahl grob nur `200` Episoden pro Welt; `5000` waeren zwar naeher an Earth-only-1000, dauern aber zu lange.

==Daraus folgt als naechste Arbeitshypothese: Das Mehrwelt-Training braucht weltabhaengige Sampling-Raten, damit Erde und Venus mehr Trainingsanteil bekommen, ohne `num_episodes` auf unpraktische Werte wie `5000` zu setzen.== Eine einfache KISS-Variante waere, Earth/Venus in der Trainingswelt-Liste mehrfach auftreten zu lassen.

Kurz gesagt: Der kleine Pilot war nicht unfaehig, er hatte nur noch nicht genug Flugstunden und einen zu engen Stundenplan.

## Trainingsdurchsatz

Empirisch aus 103 Trials auf einer L4 mit 20 parallelen Environments:

\[
\text{Env-Steps/s} \approx 851
\left(\frac{\text{optimize\_every}}{2}\right)^{0.854}
\]

`optimize_every` erklärt nahezu die gesamte Variation des Durchsatzes. Größere Werte bedeuten weniger Optimizer-Updates und damit mehr Environment-Steps pro Sekunde.

| `optimize_every` | Gemessener Median Env-Steps/s |
|---:|---:|
| 2 | 858 |
| 4 | 1.483 |
| 8 | 2.666 |

Ein einfaches Kostenmodell schätzt etwa 0,15 ms für einen Environment-Step und 2,03 ms für Backpropagation plus Optimizer-Update. Ein Update dauert also rund 14-mal so lange wie ein Environment-Step.

Die vollständige Vier-HP-Regression erreicht etwa 1,1 % mittleren relativen Fehler, aber `batch_size`, `num_episodes` und `learning_starts` liefern nur geringe zusätzliche Erklärung. Die einfache Formel ist daher vorzuziehen.

## Modellqualität hängt stark vom Trainingsseed ab

==Gleiche HPs erzeugen je nach Trainingsseed sehr unterschiedlich gute Modelle.== In der Elise-Studienreihe erreichten einzelne Trials einen mittleren Score von etwa 167 über alle fünf Welten (auf der Erde: 237,5). Die robuste Neuauswertung derselben HPs erreichte dagegen nur 113 beziehungsweise 92. Gute HPs sind daher ein erster Filter, kein Ersatz für das Speichern eines konkret guten Modells.
