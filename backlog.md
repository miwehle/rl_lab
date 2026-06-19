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
