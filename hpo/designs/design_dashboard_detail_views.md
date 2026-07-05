# Dashboard Detail Views

## Ziel

Spaeter soll das Dashboard neben dem Overview-Modus auch Detailansichten fuer einzelne Panels anbieten.

Dieses Design ist absichtlich nachgelagert. Der naechste ICRE-Schritt bleibt ein reiner Overview-Plot ohne Zoom-in.

## Motivation

In Colab kann waehrend einer laufenden HPO-Zelle keine zweite Codezelle ausgefuehrt werden.

Wenn man waehrend des HPO-Laufs von der Uebersicht in eine groessere Ansicht wechseln will, muss die Umschaltung innerhalb der bestehenden Dashboard-Ausgabe passieren.

## Grundidee

Das Dashboard bekommt zwei Ebenen:

```text
Overview        alle Panels klein, Story auf einen Blick
Detail View     ein ausgewaehltes Panel gross
```

Die Umschaltung soll moeglichst browserseitig passieren, also ohne Python-Callback und ohne neue Berechnung beim Klick.

Plotly-Buttons oder `updatemenus` sind dafuer der bevorzugte erste Weg.

## Moegliche Detailansichten

Checkpoint Robustness:

```text
candidate overview
selected checkpoint: quantiles by world
selected checkpoint: heatmap by world
selected checkpoint: summary table
```

Study:

```text
trial score timeline
best-score timeline
hover details for HPs
```

Current Trial Training:

```text
episode returns
world-colored episode returns
epsilon
trailing means
checkpoint threshold
```

## ICRE Detail

Fuer Checkpoint Robustness ist der Overview die Kandidatenentscheidung.

Die Detailansicht beantwortet danach:

```text
Warum ist dieser Checkpoint robust oder verletzlich?
Welche Welten ziehen ihn runter?
Ist das Problem Ausreisser, Streuung oder ein niedriger Median?
```

Geeignete Detailvarianten:

1. Quantile-/Interval-Plot pro Welt.
2. Heatmap der Score-Verteilung pro Welt.
3. Kleine Summary-Tabelle mit `episodes`, `mean`, `std`, `min`, `q05`, `median`, `q95`, `max`.

## Datenanforderung

Detailansichten brauchen mehr Daten als der Overview.

Fuer ICRE:

```text
Overview: scores per candidate
Detail: scores per candidate, world, episode
```

Diese Daten sollten vorab in der Figure vorhanden sein oder als Artefakt/DB-Daten bereits geladen sein. Button-Klicks duerfen im ersten interaktiven Design keine neue Evaluation starten.

## Nichtziele

- Kein Detailmodus im naechsten ICRE-Schritt.
- Keine Python-Callbacks als erster Colab-Weg.
- Keine Nachberechnung beim Klick.
- Kein grosses Widget-Framework, solange Plotly-Buttons reichen.
