## Entwurf: HPO Study Series 2A und 2B – XLander

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
  - `xlander_8d.db`
  - `xlander_11d.db`

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

### Hauptaufgabe und Vergleich

Die HPO sucht Hyperparameter für einen XLander, der auf allen vier Himmelskörpern einen hohen Gym-Score erreicht und dafür möglichst wenig Trainingsaufwand benötigt. Maßgeblich ist der gemeinsame **Quality-Effort Score**.

Series 2A und 2B beantworten zusätzlich:

> Wie stark profitiert der XLander davon, Gravitation und Wetterparameter ausdrücklich zu kennen?

Series 2A ist der robuste **Blindflug-XLander**, Series 2B der professionelle **Wetterdienst-XLander**. 🙂