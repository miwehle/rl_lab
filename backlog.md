# Backlog

## Priorisierung

| Nr. | Item | Voraus-<br>setzungen | Nutzen | Aufwand | Prio | Cool |
|---:|---|---|---|---|---|---:|
| 3 | The KISS SolarSystemLander – Quest II | 1 |  | S | High |  |
| 5 | Inside the SolarSystemLander | 1 |  | M | Med | 😎 |
| 1 | Study Series 2 |  |  | M | Med | 😎 |
| 7 | The High-Performance SolarSystemLander | 1 |  | M | Med |  |
| 2 | Austauschbare Replay-Sampling-Strategien | 1 |  | M | Med |  |
| 4 | Council of Lander Masters | 1 |  | M | Low | 😎 |

==Die Aufwände und Prios sind zu überprüfen, stimmen großteils wohl nicht.==

## 1. Study Series 2

**Ziel:** Hyperparameter für einen SolarSystemLander finden, der auf Mond, Merkur, Mars, Erde und Venus möglichst gut und mit möglichst wenig Trainingsaufwand landet.

Series 2A trainiert mit der klassischen 8D-Observation, Series 2B zusätzlich mit Gravitation, Wind- und Turbulenzparametern. Der kompakte Entwurf steht in `hpo/designs/design3.md`.

**Nutzen:** Ein gemeinsamer, wirtschaftlich trainierter Lander für fünf Himmelskörper und ein direkter Vergleich von Blindflug und Wetterdienst.

## 2. Austauschbare Replay-Sampling-Strategien

**Idee:** Das uniforme Sampling des `VectorTrainer` durch eine austauschbare Sampling-Strategie kapseln.

Mögliche Strategien:

- Uniform Experience Replay – aktueller Standard
- Prioritized Experience Replay (PER)
- reward-basiertes Downsampling
- „Trauma Filtering“ – frühe Abstürze seltener wiedergeben

**Implementierungsidee:** Das Strategy Pattern verwenden. Der Replay Buffer delegiert die Auswahl der Samples an eine übergebene Sampling-Strategie. So bleibt der `VectorTrainer` unverändert und es werden keine abgeleiteten Trainer-Klassen pro Verfahren benötigt. Uniformes Sampling bleibt der einfache Standard.

**Nutzen:** Höhere Lerneffizienz und damit möglicherweise ein besserer Quality-Effort Score.

**Voraussetzung:** Study Series 2 mit uniformem Replay abschließen.

## 3. The KISS SolarSystemLander – Quest II

*"How Small Can a Five-World Pilot Be? 🙂"*

**Frage:** Reichen kleinere Hidden Layers für einen guten SolarSystemLander?

```text
hidden_size = [32, 64, 128]
```

Beide Hidden Layers verwenden dieselbe Größe. Zunächst Landequalität und Modellgröße getrennt darstellen, statt das Quality-Effort Scoring um Modellkomplexität zu erweitern.

**Nutzen:** Ein deutlich kleineres DQN bei möglichst gleicher Landequalität.

**Voraussetzung:** Study Series 2 abschließen.

## 4. Council of Lander Masters

*"Five Masters Train One Tiny Pilot. 🙂"*

**Idee:** Je einen spezialisierten Teacher pro Himmelskörper trainieren und deren per Softmax vergleichbar gemachte Q-Werte mittels Multi-Teacher Policy Distillation auf einen kleinen SolarSystemLander übertragen.

**Nutzen:** Wissen spezialisierter Lander in einem gemeinsamen kleinen Piloten bündeln und dessen Landequalität verbessern.

**Voraussetzung:** Study Series 2 abschließen.

Weniger KISS als gemeinsames Training mit einem Replay Memory.

## 5. Inside the SolarSystemLander

*"What Does the Tiny Pilot Think? 🙂"*

**Idee:** Das Landevideo um eine Live-Visualisierung des DQN ergänzen. Neben dem Lander leuchten die Aktivierungen der beiden Hidden Layers pro Simulationsschritt auf; die vier Q-Werte werden als Balken dargestellt und die gewählte Aktion wird hervorgehoben.

Neuronen anhand ähnlicher Aktivierungsprofile räumlich gruppieren, damit mögliche Funktionsgruppen wie Sinkflug, Driftkorrektur, Lagekontrolle oder Bodennähe sichtbar werden. Ablation einzelner Neuronen oder Gruppen kann später prüfen, ob eine vermutete Funktion kausal ist.

**Implementierungsidee:** Beim Rendern eines unveränderten trainierten Modells die beiden ReLU-Aktivierungen und Q-Werte aufzeichnen und zusammen mit den Environment-Frames zu einem Video zusammensetzen.

**Nutzen:** Anschaulicher Einblick in die interne Arbeitsweise des kleinen DQN und mögliche Entdeckung interpretierbarer Neuronen- oder Funktionsgruppen.

**Voraussetzung:** Study Series 2 abschließen.

## 6. HPO-Diagnosechecks

Optuna beantwortet, welche Hyperparameter innerhalb der vorgegebenen Versuchsbedingungen am besten sind. Automatisierte Diagnosechecks sollen zusätzlich hinterfragen, ob Suchraum und Versuchsaufbau selbst plausibel waren.

Sinnvolle Hinweise:

- Das Training endet noch mit einem deutlichen Aufwärtstrend; `num_episodes` könnte zu niedrig sein.
- Der beste Hyperparameter liegt am Rand seines Suchraums; der Suchraum könnte zu eng sein.
- Das Replay Memory war immer voll oder überwiegend leer; seine Kapazität könnte unpassend sein.
- Die Exploration liegt am Trainingsende noch deutlich über `eps_end`; das Training könnte zu früh enden.
- Ein guter Gesamtscore verdeckt eine deutlich schwächere Welt.
- Die Robustness Evaluation zeigt eine hohe Streuung zwischen Trainings-Seeds.

Als erster KISS-Check wird nach einer Studie die Trainingskurve des Gewinners betrachtet: Liegt der mittlere Return der letzten 50 Episoden deutlich über dem der vorherigen 50 Episoden, erscheint ein konkreter Hinweis mit beiden Messwerten und der Empfehlung, `num_episodes` zu prüfen.

Die Checks geben Hinweise, treffen aber keine automatischen Entscheidungen. Sie sollen wenige belastbare und konkrete Meldungen erzeugen, keinen Warnungsteppich für statistische Schwankungen.

Design: hpo\designs\design_hpo_checks.md

## 7. The High-Performance SolarSystemLander

*"Five Worlds. No Economy Class. 🙂"*

**Idee:** Den kleinen 8D-Generalisten konsequent auf Landequalität trimmen. Der Gym Score zählt; längere Ausbildung und höherer Trainingsaufwand sind erlaubt.

**Ziel:** Ein robuster Mean Gym Score von mindestens 220 über alle fünf Welten.

**Einfachster Erfolgsweg:** Mit etwas Glück genügen die bisher besten robusten HPs als Ausgangspunkt, `quality_weight = 0.95` (in der Formel des QE-Scores) und eine weitere lokale Study Series mit höherem Trainingsbudget.

Design: `hpo/designs/design5.md`

## 8. Planet Lander-Programm

> **Ziel:** Einen Lander effizient entwickeln und trainieren, der auf den inneren Planeten einen Gym-Score über 200 zuverlässig erreicht.


Das Programm umfasst:

- Entwicklungsmodelle als gespeicherte Checkpoints
- HPO-unterstützte Testkampagnen
- Qualifikationstests auf allen fünf Welten
- Flugerprobung mit zusätzlichen Seeds


## 9. HPO-unterstütze Modellfolge zum multiplanetaren Gym-Score 200+

**Beobachtung:**
Einzelne hervorragende Modelle entstehen, aber erneutes Training derselben HPs streut erheblich.

**Ansatz:**
HPO erzeugt pro Trial eine Checkpoint-Folge. Aus ihr wird das beste konkrete Modell ausgewählt und samt HPs, Seed und Trainingszeitpunkt gespeichert.

**TODO:**
Speicherung konkreter guter Modelle entwerfen.


## 10. Gym Score 200+ auf allen inneren Planeten

Ansätze:
- Curriculum Learning: Heißt bei uns: Nachhilfe auf dem Mond
- 1 Modell pro Welt


## Über dieses Backlog

### Struktur der Items

Das erste Feld beschreibt Ursprung oder Charakter des Items:

- **Problem:** Ein bestehender Zustand soll verbessert werden.
- **Frage:** Eine offene Forschungsfrage soll beantwortet werden.
- **Idee:** Eine mögliche Erweiterung oder ein Experiment.
- **Hypothese:** Eine konkrete Vermutung soll geprüft werden.

Danach folgen **Nutzen** und, falls erforderlich, **Voraussetzung**.
