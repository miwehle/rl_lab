## Ziel
Mehr greedy eval score pro L4-Wandzeit.

==Metrik/Diagramm==:
* x-Achse = ==wall-clock== time auf L4
* y-Achse = ==eval score== (epsilon=0)

Marken:
* 200 = Nissan-LL / solved / gutes ROI-Ziel
* ==250== = Ferrari-LL / teueres Qualitätsziel

## Fortschritt
Kurve wandert nach oben links:
* gleicher Score schneller
* oder mehr Score bei gleicher Zeit

## Nächste Iterationen

1. ==HPO== mit aktuellem ==VectorTrainer==
   Erst vorhandenen 3x-Speedup ausnutzen.

2. ==Double DQN== fair nachtesten
   Mit eigener HPO, nicht mit HPs vom einfachen DQN.

3. ==PER== als nächste Codeänderung
   Vor allem im VectorTrainer, weil dort Replay kompakt/indexbasiert ist.

PER = Prioritized Experience Replay