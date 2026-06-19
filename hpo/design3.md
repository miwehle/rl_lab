## Entwurf: HPO Study Series 2A und 2B – SolarSystemLander

### Ziel

Ein gemeinsames DQN-Modell soll auf Mond, Mars, Erde und Venus möglichst gut landen können – trotz unterschiedlicher Gravitation und wechselnder Wetterbedingungen. Die HPO sucht dafür Hyperparameter, die hohe Landequalität mit möglichst geringem Trainingsaufwand verbinden.

### Zwei parallele Studienreihen

- **Series 2A – 8D:** klassische LunarLander-Observation  
  Der Agent muss Gravitation und Wetter indirekt aus seiner Bewegung erkennen.
- **Series 2B – 11D:** zusätzlich `gravity`, `wind_power` und `turbulence_power`  
  Der Agent landet gewissermaßen mit Wettervorhersage und Flughandbuch.

Beide Reihen verwenden ansonsten denselben Aufbau, Suchraum und dieselben Seeds.

### Parallelisierung und Speicherung

- L4 #1 führt Series 2A aus.
- L4 #2 führt Series 2B aus.
- Jede Reihe erhält eine eigene SQLite-Datenbank:
  - `solar_system_lander_8d.db`
  - `solar_system_lander_11d.db`

Damit laufen beide Reihen unabhängig und ohne konkurrierende SQLite-Zugriffe.

### Training

Jeder Trial trainiert **ein gemeinsames Q-Netz** auf allen vier Himmelskörpern:

- Körper beziehungsweise Umgebungen werden ausgewogen gemischt.
- Alle Erfahrungen gelangen in einen gemeinsamen Replay-Speicher.
- Die Replay-Memory-Größe bleibt Bestandteil der HPO.
- Kein sequenzielles Training pro Körper, um Vergessen vorheriger Welten zu vermeiden.
- Keine Teacher-Modelle oder Distillation: KISS.

### Umgebungen

| Körper | `gravity` | beispielhafte `wind_power`-Verteilung | `turbulence_power`-Verteilung |
|---|---:|---:|---:|
| Mond | −1.65 | 0 | 0 |
| Mars | −3.8 | U(0, 4) | U(0, 1) |
| Erde | −10 | U(5, 15) | U(0, 2) |
| Venus | −9 | U(15, 20) | U(0, 2) |

Die Wetterparameter werden reproduzierbar anhand des Seeds pro Episode gezogen. Die Intervalle sind plausible Szenariodefinitionen, keine exakten physikalischen Modelle.

Gymnasium erzeugt daraus zusätzlich den zeitlich wechselnden linearen Wind und das Drehmoment.

### Baseline und Bewertung

- Ausgangskonfiguration: Gewinner-Hyperparameter aus Study Series 1.
- Diese Konfiguration wird für die neue Vier-Körper-Aufgabe erneut trainiert und bewertet.
- Bewertung ausgewogen über alle vier Körper und definierte Wetter-Seeds.
- Optuna verwendet den bereits entworfenen **Quality-Effort Score**.
- Neben dem Gesamtscore sollten die Gym-Scores je Körper gespeichert werden, damit ein guter Durchschnitt keine Schwäche auf einer einzelnen Welt verdeckt.

### Suchräume in den Studien

Series 2A und 2B verwenden dieselben Suchräume. S0 übernimmt die Gewinner-Hyperparameter aus Study Series 1 als neue Baseline; `num_episodes` kommt als zusätzlicher HP hinzu.

| HP | S0 Baseline | S1 Update-Ökonomie | S2 Exploration | S3 Replay-Kapazität | S4 Gemeinsame Feinsuche |
|---|---|---|---|---|---|
| learning_rate | 0.001606 | *float(5e-4, 3e-3, log=True)* | best(S1) | best(S1) | *float(best / 2, best * 2, log=True)* |
| batch_size | 1_024 | *categorical([256, 512, 1_024])* | best(S1) | best(S1) | *categorical(neighbors(best(S1), [256, 512, 1_024]))* |
| eps_end | 0.0197 | 0.0197 | *float(0.01, 0.10)* | best(S2) | *float(max(0.01, best - 0.02), min(0.10, best + 0.02))* |
| eps_decay | 8_446 | 8_446 | *int(5_000, 100_000, log=True)* | best(S2) | *int(best / 2, best * 2, log=True)* |
| gamma | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| tau | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| learning_starts | 2_500 | *categorical([1_000, 2_500, 5_000])* | best(S1) | best(S1) | *categorical(neighbors(best(S1), [1_000, 2_500, 5_000]))* |
| optimize_every | 4 | *categorical([2, 4, 8])* | best(S1) | best(S1) | *categorical(neighbors(best(S1), [2, 4, 8]))* |
| replay_memory_capacity | 200_000 | 200_000 | 200_000 | *categorical([100_000, 200_000, 400_000])* | best(S3) |
| num_episodes | 600 | *categorical([500, 1_000, 1_500, 2_000])* | best(S1) | best(S1) | *categorical(neighbors(best(S1), [500, 1_000, 1_500, 2_000]))* |

Notation:

- *kursiv*: Wert wird in dieser Studie von Optuna gewählt.
- ohne Markierung: Wert bleibt fest oder wird aus einer vorherigen Studie übernommen.
- `float`, `int` und `categorical` stehen für `trial.suggest_float`, `trial.suggest_int` und `trial.suggest_categorical`.
- `best(Sx)`: bester Wert aus Studie x.
- `neighbors(b, M)`: b plus direkte Nachbarn in Menge M.

### Hauptaufgabe und Vergleich

Die HPO sucht Hyperparameter für einen SolarSystemLander, der auf allen vier Himmelskörpern einen hohen Gym-Score erreicht und dafür möglichst wenig Trainingsaufwand benötigt. Maßgeblich ist der gemeinsame **Quality-Effort Score**.

Series 2A und 2B beantworten zusätzlich:

> Wie stark profitiert der SolarSystemLander davon, Gravitation und Wetterparameter ausdrücklich zu kennen?

Series 2A ist der robuste **SolarSystemLander im Blindflug**, Series 2B der professionelle **SolarSystemLander mit Wetterdienst**. 🙂
