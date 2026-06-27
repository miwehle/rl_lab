# HPO Insights

## SolarSystemLander: 8D, 11D, naechster 9D-Versuch

Die bisherigen Elise-Studien sprechen dafuer, dass 8D trotz fehlendem Weltkontext noch staerker ist als 11D. Ein 11D-Lauf (`solar_system_lander_11d_elise.db`) erreichte nur etwa `92` als besten Trial-Score und etwa `48` robust. Das Muster war auffaellig: Mond, Merkur und Mars wurden oft ordentlich geflogen, Erde und Venus blieben meist schlecht.

Die 11D-Zusatzsignale `wind_power` und `turbulence_power` sind vermutlich nicht so sauber wie `gravity`: sie beschreiben episodische Wetterparameter, nicht die momentan wirkende Kraft. Dadurch koennen sie dem kleinen Netz mehr Verwirrung als Hilfe geben.

8D zeigte mehr Potenzial. Ein 211er wurde in einer fruehen Robustness-Evaluation beobachtet, war aber nicht als Modellcheckpoint gesichert. Der beste noch gesicherte Pilot ist der 8D-Elise-bunt-Checkpoint `trial_0009_eval_best.pt` mit Score `180.31` ueber alle fuenf inneren Welten.

Die Epsilon-Region um `eps_end ~= 0.05..0.06` und `eps_decay ~= 40_000..80_000` kann starke 8D-Einzelpiloten erzeugen: eine fokussierte S3-Folgestudie fand dort erneut einen Kandidaten um `175`. Einzelne HP-Saetze bleiben aber stark seed-abhaengig, und 8D bleibt strukturell fragwuerdig, weil dem Netz der Weltkontext fehlt.

Beste gefundene 8D-HPs fuer konkrete Spitzenpiloten:

| Quelle | Score | `eps_end` | `eps_decay` | Hinweis |
|---|---:|---:|---:|---|
| `8d_elise_bunt`, Trial 9 | 180.31 | 0.055604 | 67181 | bestes gesichertes Modell |
| `8d_elise_optevery2`, Trial 18 | 153.54 | 0.038376 | 34218 | bester Trial in dieser DB, robust etwa 105 |

Die eigentliche Arbeitshypothese liegt nun auf 9D: originale 8D-LunarLander-Observation plus `gravity / 12`, aber ohne Wind- und Turbulenzparameter. Das koennte der Sweet Spot sein: genug Weltkontext fuer das Netz, aber keine fragwuerdigen Zusatzsignale.

Fuer eine erste 9D-Studie ist ein moderat breiter Suchraum plausibel:

```python
trial.suggest_float(HP.EPS_END, 0.045, 0.065)
trial.suggest_int(HP.EPS_DECAY, 40_000, 80_000, log=True)
```

Der konkrete 180er bleibt wichtig: Er beweist, dass das kleine Netz bereits einen brauchbaren Innenplaneten-Piloten erzeugen kann. Die HPO-Aufgabe ist nun, solche Piloten zuverlaessiger und robuster zu produzieren.

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

Gleiche HPs erzeugen je nach Trainingsseed sehr unterschiedlich gute Modelle. In der Elise-Studienreihe erreichten einzelne Trials einen mittleren Score von etwa 167 über alle fünf Welten (auf der Erde: 237,5). Die robuste Neuauswertung derselben HPs erreichte dagegen nur 113 beziehungsweise 92. Gute HPs sind daher ein erster Filter, kein Ersatz für das Speichern eines konkret guten Modells.
