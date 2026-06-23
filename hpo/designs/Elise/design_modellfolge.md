## Essenz des Designs

Ausgangsmodell → Trainingsabschnitt → Checkpoint-Folge
                                  → bestes Modell sichern

==checkpoint-based HPO== (lineage-based HPO)


## Allgemeines zu HPO

HPO-Arten
1. conventional HPO (independent-trial HPO)
2. checkpoint-based HPO (lineage-based HPO)

zu 2:
Elternmodell + neue HPs → Nachfolgemodelle → bester Checkpoint

==Hybrid==:
mehrere Nachfolger eines guten Checkpoints trainieren
→ jeweils besten Checkpoint speichern
→ auf allen Welten qualifizieren
→ bestes Modell wird Elternmodell der nächsten Iteration


## Ausführlicheres Design

==Iterativ-optimierend==

Jede Iteration beginnt mit dem bisher besten konkreten Modell und soll ein besseres Entwicklungsmodell hervorbringen.

```text
Entwicklungsmodell
        ↓
HPO-Trainingsabschnitte
        ↓
Checkpoint-Folgen
        ↓
beste Checkpoints auswählen
        ↓
Qualifikation auf fünf Welten
        ↓
neues Entwicklungsmodell
```

In jeder Iteration:

1. **Ausgangsmodell festlegen**  
   Der bisher beste Checkpoint ist der gemeinsame Startpunkt.

2. **Trainingsabschnitte ausprobieren**  
   Optuna schlägt HPs vor. Jeder Trial trainiert eine Kopie des Ausgangsmodells beispielsweise 200 Episoden weiter.

3. **Checkpoints speichern**  
   Während jedes Trials werden Zwischenstände gesichert und bewertet.

4. **Trial-Kandidaten auswählen**  
   Pro Trial bleibt der beste konkrete Checkpoint erhalten, nicht bloß dessen HPs.

5. **Kandidaten qualifizieren**  
   Die besten Modelle werden auf allen fünf Welten mit zusätzlichen Evaluationsseeds geprüft. ~~Maßgebend ist zunächst die schwächste Welt, damit gute Erde-Werte keinen schlechten Mond ausgleichen.~~

6. **Entwicklungsmodell aktualisieren**  
   Nur ein besser qualifiziertes Modell ersetzt den bisherigen Stand. Andernfalls bleibt der alte Checkpoint erhalten.

7. **Ziel prüfen**  
   Erreicht das Modell zuverlässig über 200 auf jeder Welt, ist es qualifiziert. Sonst beginnt die nächste Iteration.


Der entscheidende Unterschied zur bisherigen HPO:

```text
bisher: HPs → immer wieder neue Modelle

neu:     Modell → verbesserte Nachfolgemodelle
```

Das ist tatsächlich eine **Model Lineage**: Jede Iteration erzeugt Nachfolger eines bekannten Entwicklungsmodells.


## Kleinster nächster Schritt

Zuerst speichern wir pro Trial den besten konkreten Checkpoint samt Score, HPs, Seed und Episode. Noch keine große Lineage-Verwaltung, kein Replay-Speicher, keine komplexe Artefaktstruktur.

## Mögliche Erweiterungen

### Modellbaum

Zunächst wird nur eine Modelllinie weiterentwickelt. Eine spätere Erweiterung zu einem kleinen Modellbaum ist möglich, falls gute alternative Linien zu früh verloren gehen.