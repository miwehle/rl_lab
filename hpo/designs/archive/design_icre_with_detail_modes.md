# Integrated Checkpoint Robustness Eval

## Ziel

Das Dashboard soll Checkpoint Robustness Evaluation als eigenes HPO-Kapitel erzaehlen: Ist der konkrete gespeicherte Pilot wirklich robust, auf welchen Welten ist er stark, und wo ist er verletzlich?

ICRE ersetzt nicht HP Robustness. HP Robustness fragt nach guten Hyperparametern, ICRE fragt nach Vertrauen in einen konkreten Checkpoint.

## Ausgangslage

`evaluate_checkpoint_robustness(...)` ist bereits live ins Dashboard anschliessbar, nutzt dort aber nur den generischen Robustness-Plot: Candidate, Source Score, Robust Eval und Mittelwert.

Die aussagekraeftigen Checkpoint-Qualifikationsplots liegen bisher in separaten Notebook-Zellen: `checkpoint_scores(...)`, `score_summary(...)`, `plots.heatmap(...)` und `plots.quantiles(...)`.

Das ist fuer Colab unguenstig: Waehrend die HPO-Zelle laeuft, kann keine zweite Codezelle ausgefuehrt werden. Umschalten zwischen Overview und Detailansicht muss daher innerhalb der laufenden Dashboard-Ausgabe interaktiv gehen.

> [!NOTE] Design CR
> Erstmal bauen wir keine Detailansicht ein.

## Dashboard Story

Das Dashboard bleibt der HPO-Geschichtenerzaehler:

```text
Study Series                 Wo steht die Serie?
Current HPs                  Welche Parameter gelten gerade?
Study                        Was passiert in der aktuellen Study?
HP Robustness                Welche HP-Kandidaten wirken belastbar?
Checkpoint Qualification     Ist der konkrete Pilot robust?
Current Trial Training       Wie lernt der laufende Trial?
```


> [!NOTE] Design CR
> HP Robustness Eval kommt erstmal raus aus dem Dashboard. Im Dashboard ist nicht viel Platz, es soll übersichtlich bleiben. Checkpoint Robustness Eval ist wichtiger bzw. zielführender für uns.

Im Uebersichtsmodus sollen alle Panels sichtbar bleiben. ICRE soll dort kompakt zeigen, ob der Pilot ueber alle Welten qualifiziert wirkt.

Im Fokusmodus darf ein Panel die ganze Figure einnehmen. Fuer ICRE kann der Fokusmodus zwischen Varianten umschalten.

## Plotwahl

Default fuer ICRE ist der Quantile-Plot pro Welt.

> [!NOTE] Design CR
> Weniger. Im Dashboard (im Überblick-Modus) soll für ICRE ein Plot über alle 3 Kandidaten drin sein. Sonst für ICRE im Überblick-Modus erstmal nix. Evtl. interaktives Umschalten auf ICRE-Detail-Modus ("Zoom in"). Erstmal lassen wir das Dashboard nur im Overview-Modus. Detail-Modus (für ein jeweils ausgewähltes Panel) kann später hinzukommen.


Er zeigt kompakt:

- `min..max`
- `q05..q95`
- `q25..q75`
- `median`
- `mean`

Der Quantile-Plot ist fuer die Dashboard-Story besser als die Heatmap, weil er Robustheit und schwache Welten sofort sichtbar macht.

Die Heatmap bleibt wichtig als Fokus-Variante, wenn man die genaue Score-Verteilung sehen will.

## Interaktivitaet in Colab

Die Umschaltung soll moeglichst browserseitig passieren.

Keine neue Notebook-Zelle soll noetig sein, solange HPO laeuft.

KISS-Start:

```text
Overview
Focus: Study Series
Focus: Study
Focus: HP Robustness
Focus: Checkpoint Qualification
Focus: Current Trial Training
```

Fuer `Focus: Checkpoint Qualification`:

```text
variant: quantiles
variant: heatmap
variant: summary
```

Die erste Implementierung darf diese Modi als Plotly-`updatemenus`/Buttons bauen, solange dabei nur vorhandene Traces sichtbar/unsichtbar geschaltet werden. Keine Python-Callbacks und keine Nachberechnung beim Klick.

## Datenfluss

ICRE braucht aggregierte Daten fuer die Uebersicht und optional Rohscores fuer Fokusvarianten.

> [!NOTE] Design CR
> Wie schon geschrieben: Erstmal keine weiteren Zoom-In-Stufen bzw. Fokusvarianten. Bitte mache aus diesem Design zwei: 1) Nächster Schritt (ICRE als übersichtlichen  Quantilen-Plot ins Dashboard integrieren). 2) Interaktives Umschalten in Detail-Ansichten.

```text
checkpoint_scores(...) -> scores DataFrame
score_summary(scores)  -> summary DataFrame
dashboard              -> CheckpointQualificationData
```

Moegliche kleine Datenklasse:

```python
@dataclass(frozen=True)
class CheckpointQualification:
    checkpoint_path: str
    episodes: int
    scores: pd.DataFrame
    summary: pd.DataFrame
```

Fuer den Uebersichtsmodus reicht `summary`. Fuer Heatmap und Hoverdetails braucht der Fokusmodus `scores`.

## API-Skizze

Kleinster sinnvoller Einstieg:

```python
dashboard.report_checkpoint_qualification(qualification)
```

oder beim Figure-Bau:

```python
build_dashboard(..., checkpoint_qualification=qualification)
```

Die Notebook-Zelle, die `checkpoint_scores(...)` berechnet, kann zunaechst bestehen bleiben. Sie uebergibt danach die Daten an das Dashboard.

Spaeter kann `evaluate_checkpoint_robustness(...)` optional selbst eine Qualification erzeugen, wenn die Kosten dafuer akzeptabel sind.

## Umsetzungsschritte

1. Eine kleine `CheckpointQualification`-Datenklasse einfuehren.
2. Plotly-Version des Quantile-Plots fuer `summary` bauen.
3. ICRE als sechstes Dashboard-Panel in den Overview integrieren.
4. Fokusmodus fuer mindestens `Overview` und `Checkpoint Qualification` entwerfen.
5. Heatmap erst danach als Fokus-Variante hinzufuegen.

## Nichtziele fuer den ersten Schritt

- Kein vollstaendiges interaktives Widget-System.
- Keine Python-Callbacks in Colab.
- Keine neue Evaluation waehrend eines Button-Klicks.
- Keine Vermischung von HP Robustness und Checkpoint Qualification.
- Kein Versuch, alle Notebook-Analyseplots sofort ins Dashboard zu ziehen.
