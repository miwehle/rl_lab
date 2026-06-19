# Backlog

## Austauschbare Replay-Sampling-Strategien

**Idee:** Das uniforme Sampling des `VectorTrainer` durch eine austauschbare
Sampling-Strategie kapseln.

Mögliche Strategien:

- Uniform Experience Replay – aktueller Standard
- Prioritized Experience Replay (PER)
- reward-basiertes Downsampling
- „Trauma Filtering“ – frühe Abstürze seltener wiedergeben

**Implementierungsidee:** Das Strategy Pattern verwenden. Der Replay Buffer
delegiert die Auswahl der Samples an eine übergebene Sampling-Strategie. So
bleibt der `VectorTrainer` unverändert und es werden keine abgeleiteten
Trainer-Klassen pro Verfahren benötigt. Uniformes Sampling bleibt der einfache
Standard.

**Erwarteter Nutzen:** Höhere Lerneffizienz und damit möglicherweise ein
besserer Quality-Effort Score.

**Priorität:** Später untersuchen; Study Series 2 zunächst mit uniformem Replay.

## The KISS SolarSystemLander – Quest II

*"How Small Can a Five-World Pilot Be? 🙂"*

**Frage:** Reichen kleinere Hidden Layers für einen guten SolarSystemLander?

```text
hidden_size = [32, 64, 128]
```

Beide Hidden Layers verwenden dieselbe Größe. Zunächst Landequalität und Modellgröße getrennt darstellen, statt das Quality-Effort Scoring um Modellkomplexität zu erweitern.

**Erwarteter Nutzen:** Ein deutlich kleineres DQN bei möglichst gleicher Landequalität.

**Priorität:** Nach Study Series 2 untersuchen.

## Council of Lander Masters

*"Five Masters Train One Tiny Pilot. 🙂"*

**Idee:** Je einen spezialisierten Teacher pro Himmelskörper trainieren und deren per Softmax vergleichbar gemachte Q-Werte mittels Multi-Teacher Policy Distillation auf einen kleinen SolarSystemLander übertragen.

**Priorität:** Spätere Quest; weniger KISS als gemeinsames Training mit einem Replay Memory.

## Inside the SolarSystemLander

*"What Does the Tiny Pilot Think? 🙂"*

**Idee:** Das Landevideo um eine Live-Visualisierung des DQN ergänzen. Neben dem Lander leuchten die Aktivierungen der beiden Hidden Layers pro Simulationsschritt auf; die vier Q-Werte werden als Balken dargestellt und die gewählte Aktion wird hervorgehoben.

Neuronen anhand ähnlicher Aktivierungsprofile räumlich gruppieren, damit mögliche Funktionsgruppen wie Sinkflug, Driftkorrektur, Lagekontrolle oder Bodennähe sichtbar werden. Ablation einzelner Neuronen oder Gruppen kann später prüfen, ob eine vermutete Funktion kausal ist.

**Implementierungsidee:** Beim Rendern eines unveränderten trainierten Modells die beiden ReLU-Aktivierungen und Q-Werte aufzeichnen und zusammen mit den Environment-Frames zu einem Video zusammensetzen.

**Erwarteter Nutzen:** Anschaulicher Einblick in die interne Arbeitsweise des kleinen DQN und mögliche Entdeckung interpretierbarer Neuronen- oder Funktionsgruppen.

**Priorität:** Nach Study Series 2 untersuchen.
