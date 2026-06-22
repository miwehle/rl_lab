## HPO Study Series 4 – SolarSystemLander 8D „Porsche“

### Big Five aus den bisherigen Studienreihen

1. **Der 8D-Generalist ist bisher stärker als der 11D-Lander mit Wetterbericht.** Die drei zusätzlichen Eingaben aus Gravitation, Wind und Turbulenz brachten keinen Vorteil; Study Series 2A erreichte robust Gym `111,7`, Series 2B Gym `92,5`.
2. **Lokale Feinsuche hat den 8D-Lander deutlich verbessert.** Study Series 3A erreichte nach der damaligen, noch schwachen Robustness Evaluation Gym `141,9` und QE `−0,714`; der beste einzelne Trial erreichte Gym `179,4`.
3. **Der beste Trial war nach 500 Episoden noch nicht ausgelernt.** Sein mittlerer Trainingsreturn stieg von `82,3` in den Episoden 401–450 auf `134,1` in den Episoden 451–500; Epsilon lag am Ende mit `0,117` noch deutlich über `eps_end = 0,051`.
4. **Replay Memory ist für den Fünf-Welten-Generalisten Langzeitgedächtnis.** Bisher spricht das Ergebnis dafür, möglichst alle während des Trainings erzeugten Transitions zu behalten; bei längeren Trainings muss die Kapazität deshalb zusammen mit `num_episodes` wachsen.
5. **Mond und Venus bleiben die entscheidenden Prüfsteine, und die Seed-Streuung ist groß.** Der Mond liegt dynamisch weit vom mittleren Trainingsfall entfernt, Venus kombiniert hohe Gravitation mit starkem Wind; gute Einzeltrials reichen daher nicht als Nachweis und müssen über mehrere Trainings-Seeds bestätigt werden.

### Ziel

Study Series 4 soll einen „Porsche“ ausbilden: Der robuste, gleichgewichtete Mean Gym Score über Mond, Merkur, Mars, Erde und Venus soll mindestens `220` erreichen. Trainingskosten sind zweitrangig; gesucht wird zunächst die erreichbare Qualität des bestehenden 8D-Generalisten mit 256 Neuronen.

### Bewertung

- `quality_weight = 0.95`: Qualität dominiert, Trainingsaufwand bleibt nur ein kleiner Tie-Breaker.
- `quality_min = 200` und `quality_target = 250` bleiben unverändert.
- Der gemeinsame Gym Score ist der gleichgewichtete Mittelwert der fünf Welt-Scores.
- Pro Welt werden weiterhin zehn Greedy-Eval-Episoden mit festen Seeds ausgewertet.
- Erfolgskriterium ist der robuste Mean Gym Score `>= 220`, nicht ein bestimmter QE-Score.

### Robustness Evaluation

Nach jeder Studie werden die fünf besten Kandidaten beurteilt. Jeder Kandidat besteht aus seinem ursprünglichen Trainingslauf und vier zusätzlichen Trainings-Seeds, also fünf Trainingsläufen insgesamt. Kandidatenauswahl und gespeicherte robuste Scores verwenden das arithmetische Mittel; die Einzelergebnisse bleiben im Dashboard sichtbar.

### Studien

| Studie | Ziel | Trials |
|---|---|---:|
| S1 Flight Hours | Ausreichendes Trainingsbudget bestimmen | 25 |
| S2 Exploration Schedule | Exploration an das längere Training anpassen | 25 |
| S3 Learning Regime | Lernen und Optimieren für lange Trainings einstellen | 25 |
| S4 Joint Finetune | Gewinner gemeinsam fein abstimmen | 40 |

S0 übernimmt den finalen robusten Incumbent aus Study Series 3B und bewertet ihn mit fünf Trainingsläufen neu. Die Replay-Kapazität wird bereits für S0 auf `1_200_000` vergrößert, damit sie während der gesamten Reihe korrekt als fester HP geführt wird. Die 500 Baseline-Episoden schöpfen die bisherigen 400.000 Plätze ohnehin nicht aus.

### Suchräume

| HP | S0 Incumbent | S1 Flight Hours | S2 Exploration Schedule | S3 Learning Regime | S4 Joint Finetune |
|---|---:|---|---|---|---|
| learning_rate | incumbent | incumbent | incumbent | *float(incumbent × 0.5, incumbent × 1.25, log=True)* | *float(incumbent × 0.75, incumbent × 1.25, log=True)* |
| batch_size | incumbent | incumbent | incumbent | *categorical([512, 1_024])* | *categorical(neighbors(incumbent, [512, 1_024]))* |
| eps_end | incumbent | incumbent | *float(0.03, 0.08)* | incumbent | *float(max(0.02, incumbent − 0.01), min(0.10, incumbent + 0.01))* |
| eps_decay | incumbent | incumbent | *int(75_000, 300_000, log=True)* | incumbent | *int(incumbent × 0.75, incumbent × 1.25, log=True)* |
| gamma | incumbent | incumbent | incumbent | incumbent | incumbent |
| tau | incumbent | incumbent | incumbent | incumbent | incumbent |
| learning_starts | incumbent | incumbent | incumbent | *categorical([2_500, 5_000, 10_000])* | *categorical(neighbors(incumbent, [2_500, 5_000, 10_000]))* |
| optimize_every | incumbent | incumbent | incumbent | *categorical([4, 8])* | *categorical(neighbors(incumbent, [4, 8]))* |
| replay_memory_capacity | 1_200_000 | 1_200_000 | 1_200_000 | 1_200_000 | 1_200_000 |
| num_episodes | incumbent | *categorical([1_000, 1_500, 2_000])* | incumbent | incumbent | *categorical(neighbors(incumbent, [1_000, 1_500, 2_000]))* |

Notation:

- *kursiv*: Wert wird in dieser Studie von Optuna gewählt.
- `incumbent`: bisher bester robust überprüfter Wert.
- `neighbors(b, M)`: `b` plus direkte Nachbarn in der geordneten Menge `M`.
- `float`, `int` und `categorical` entsprechen `trial.suggest_float`, `trial.suggest_int` und `trial.suggest_categorical`.

### Erwartete Aussage

Erreicht der robuste Gewinner Gym `>= 220`, kann die bestehende Trainingsinfrastruktur einen leistungsfähigen Fünf-Welten-Generalisten ausbilden, sofern genügend Trainingsbudget bereitsteht. Bleiben insbesondere Mond und Venus trotz längerer Ausbildung deutlich zurück, spricht das stärker für eine Kapazitäts- oder Architekturgrenze und damit für größere Netze oder mehrere spezialisierte Policies.


### Kurz und knackig formuliert

Fünf Welten gibt es nicht zum Economy-Tarif. ;o)

Series 3B zeigt: Der kleine Generalist kann landen, aber seine Ausbildung ist noch zu kurz und stark seedabhängig. Die nächste Reihe darf deshalb investieren:

- `quality_weight = 0.95`
- 1.000–2.000 Episoden
- ausreichend großes Replay Memory
- robuste Abnahme über fünf Trainingsläufe
- Ziel: Mean Gym ≥ 220

Jetzt bauen wir keinen sparsamen Mietwagen mehr, sondern einen Piloten mit echten Flugstunden.
