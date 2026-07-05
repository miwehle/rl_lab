# Final Integrated Checkpoint Robustness Eval

## Anlass

Das bisherige `design_icre.md` hat Checkpoint Robustness ins Dashboard-Panel gebracht, aber noch nicht in den normalen HPO-Ablauf integriert.

Genauer: Checkpoint Robustness kann aktuell im Dashboard live angezeigt werden, wenn eine separate Notebook-Zelle `evaluate_checkpoint_robustness(...)` aufruft und `progress_fn=runner.reporter.report_robustness_evaluation` uebergibt.

Das ist nur halb integriert. Die Anzeige ist integriert, die HPO-Orchestrierung noch nicht.

Dieses Design beschreibt den finalen KISS-Schnitt: Checkpoint Robustness wird ein normaler Teil von `StudyRunner.run(...)`; die separate Notebook-Zelle entfaellt.

## Ziel

Nach jeder Study entscheidet nicht mehr HP Robustness ueber den Sieger, sondern Checkpoint Robustness ueber konkrete gespeicherte Piloten.

Der normale Ablauf wird:

```text
StudyRunner.run(...)
  -> Optuna-Trials trainieren
  -> eval-best Herausforderer-Checkpoints der Trials sammeln
  -> Top-N Herausforderer robust evaluieren
  -> besten robusten Checkpoint als Study-Ergebnis speichern
  -> Incumbent aktualisieren
  -> Dashboard zeigt alles live
```

Das Dashboard bleibt der HPO-Geschichtenerzaehler. Es soll nicht nur eine manuell gestartete Nachanalyse visualisieren, sondern die echte Siegerpruefung der Study.

## Begriffe

`ObjectiveConfig.eval_episodes` bleibt die schnelle Trial-Evaluation waehrend Optuna.

`StudyRunner.robust_candidates` bleibt der Name fuer die Anzahl robuster Kandidaten, bedeutet kuenftig aber Top-N Checkpoints, nicht mehr Top-N HP-Kandidaten.

`StudyRunner.robust_eval_episodes` wird neu eingefuehrt und bedeutet Eval-Episoden pro Welt und Checkpoint in der integrierten Checkpoint Robustness. Bei fuenf Welten sind das `5 * robust_eval_episodes` Landeversuche pro Checkpoint.

`extra_seeds` entfaellt aus `StudyRunner`, weil die alte HP-Robustness nicht mehr der normale Runner-Pfad ist.

## Warum Nicht Extra Seeds?

Bei HP Robustness bedeuteten `extra_seeds`: Kandidaten mit denselben HPs nochmal komplett neu trainieren.

Bei Checkpoint Robustness wird nicht neu trainiert. Der Checkpoint ist ein konkreter Pilot; Robustheit entsteht durch viele Evaluations-Episoden ueber die Welten.

Darum ist `eval_episodes` der passende Parameter fuer Checkpoint Robustness, nicht `extra_seeds`.

Die bisherige Notebook-Zelle nutzte bereits `eval_episodes=200` pro Welt. Dieser Wert soll nach Entfernen der Zelle weiterhin vom Notebook aus gesetzt werden koennen, jetzt aber am Runner.

## Runner API

Minimaler neuer Runner-Aufruf:

```python
runner = StudyRunner(
    ...,
    robust_candidates=3,
    robust_eval_episodes=50,
    sync_fn=STUDY_SERIES_STORAGE.backup,
)
```

`robust_candidates=3` waehlt die drei besten gespeicherten eval-best Herausforderer-Checkpoints nach Source Score aus.

`robust_eval_episodes=50` evaluiert jeden dieser Checkpoints mit 50 Episoden pro Welt.

Bei fuenf Welten bedeutet das:

```text
3 Checkpoints * 5 Welten * 50 Episoden = 750 Eval-Episoden
```

Das ist fuer eine Siegerpruefung deutlich billiger als erneutes Training, aber deutlich aussagekraeftiger als die schnelle Trial-Eval mit `ObjectiveConfig.eval_episodes=10`.

## StudyRunner Ablauf

`StudyRunner.run(...)` ruft nach `run_study(...)` direkt Checkpoint Robustness auf:

```python
checkpoint_results = evaluate_checkpoint_robustness(
    study=study,
    objective_cfg=objective_cfg,
    top_n=self.robust_candidates,
    eval_episodes=self.robust_eval_episodes,
    progress_fn=self.reporter.report_robustness_evaluation,
)
```

`progress_fn` bleibt der bestehende Reporter-Kanal. Dadurch muss das Dashboard keinen neuen Kanal lernen.

Die bestehende Kandidatenebenen-Granularitaet reicht erstmal: Das Panel aktualisiert sich nach Kandidat 1, Kandidat 2 und Kandidat 3.

## Herausforderer-Qualifikation

Checkpoint-Speicherung bleibt hart an den bisherigen Incumbent Score gekoppelt.

Wenn `self.incumbent_score` gesetzt ist, bleibt die bestehende Logik erhalten:

```python
objective_cfg = _with_checkpoint_min_score(
    objective_cfg,
    self.incumbent_score,
)
```

Dadurch werden nur Checkpoints gespeichert, die den bisherigen Incumbent bereits in der schnellen Trial-Evaluation schlagen.

Diese gespeicherten Checkpoints sind die Herausforderer fuer CRE.

Es gibt keinen Margin-Parameter. Ein Kandidat muss sich zuerst in der schnellen Trial-Evaluation qualifizieren und danach in CRE robust genug sein, um Incumbent zu werden.

Wenn keine Herausforderer-Checkpoints gespeichert wurden, gibt es keine CRE-Kandidaten. Dann wird `checkpoint_robustness = []` gespeichert, der Incumbent bleibt unveraendert, und die Study gilt als fertig.

Dieser Randfall ist wichtig, weil sich dort Resume-Bugs leicht absetzen koennen.

Die Semantik des Study-Attrs ist:

```text
checkpoint_robustness fehlt   -> CRE ist noch nicht erledigt
checkpoint_robustness == []    -> CRE ist erledigt, aber es gab keine Kandidaten
checkpoint_robustness hat Werte -> CRE ist erledigt und hat Kandidaten bewertet
```

Die Implementierung darf den leeren Fall daher nicht nur "still ueberspringen", sondern muss die leere Liste explizit speichern.

## Siegerregel

KISS-Regel: Unter den CRE-Kandidaten gewinnt der Checkpoint mit dem hoechsten `robust_score`.

`evaluate_checkpoint_robustness(...)` setzt `robust_score` bereits auf den Mean ueber alle gesammelten Checkpoint-Evaluationsscores.

Der Runner liest das beste Resultat aus `checkpoint_results`, nicht mehr `study.user_attrs["robust_best_score"]`.

Die weiteren Werte aus `checkpoint_summary` bleiben Berichtswerte fuer Dashboard und Analyse:

- `median`
- `min`
- `max`
- `q05`
- `q25`
- `q75`
- `q95`
- `world_scores`

Sie gehen in dieser Implementierungsstufe nicht in die Auswahlformel ein.

## Incumbent Update

Die Study soll weiterhin `incumbent_params` und `incumbent_score` speichern.

`incumbent_score` wird kuenftig aus dem besten Checkpoint-Robustness-Ergebnis genommen.

`incumbent_params` bleiben die HPs des Trials, zu dem der robuste Sieger-Checkpoint gehoert.

Dafuer muss der Runner den Trial des Gewinner-Checkpoints ueber `trial_number` finden und dessen Params in den Incumbent uebernehmen.

Wenn der robuste Sieger den bisherigen Incumbent Score nicht schlaegt, bleibt der bisherige Incumbent erhalten.

Das ist die zweite Stufe der Herausforderung:

```text
1. Qualifikation: schneller Trial-Score > Incumbent Score
2. Titelgewinn: CRE robust_score > Incumbent Score
```

## Study Attrs

`evaluate_checkpoint_robustness(...)` speichert bereits:

```python
study.set_user_attr("checkpoint_robustness", results)
```

Der Runner speichert danach wie bisher:

```python
study.set_user_attr("incumbent_params", self.incumbent_params)
study.set_user_attr("incumbent_score", self.incumbent_score)
```

`robust_best_score` und `robust_best_params` gehoeren zur alten HP-Robustness und werden vom normalen Runner-Pfad nicht mehr geschrieben.

## Finished Check

`_study_already_finished(...)` muss kuenftig auf die neue Completion-Signatur pruefen:

```text
finished trials >= n_trials
and "checkpoint_robustness" in study.user_attrs
and "incumbent_score" in study.user_attrs
```

Damit wird eine nach Optuna abgeschlossene, aber noch nicht robust gepruefte Study nicht faelschlich als fertig betrachtet.

Eine Study ohne gespeicherte Herausforderer-Checkpoints gilt erst dann als fertig, wenn `checkpoint_robustness` explizit als leere Liste gespeichert wurde.

## Notebook

Die separate Checkpoint-Robustness-Zelle entfaellt:

```python
evaluate_checkpoint_robustness(...)
```

Der Runner bekommt stattdessen die Parameter:

```python
robust_candidates=3
robust_eval_episodes=200
```

Analysezellen duerfen spaeter nur noch gespeicherte Ergebnisse lesen:

```python
study.user_attrs["checkpoint_robustness"]
```

Sie duerfen die normale Robustness nicht erneut ausloesen.

## HP Robustness

`hpo.evaluation.hp_robustness` bleibt als Modul erhalten.

Auch die Tests fuer HP Robustness bleiben erhalten.

Das Dashboard-Panel fuer HP Robustness bleibt ebenfalls erhalten. Es stoert nicht, weil es weiter ueber denselben `RobustnessProgress`-Kanal funktioniert und keinen eigenen Runner-Code braucht.

Erhalten bleibt also die Mechanik:

- `select_robust_best(...)`
- HP-Robustness-Tests
- Dashboard-Darstellung fuer HP-Robustness-Progress
- alte Progress-Datenform ueber `candidate_seed_scores`

Aus dem normalen Runner-Pfad entfernt wird die Orchestrierung:

- `StudyRunner.extra_seeds`
- automatischer `select_robust_best(...)`-Aufruf in `StudyRunner.run(...)`
- Incumbent-Logik ueber `robust_best_score` und `robust_best_params`
- `_study_already_finished(...)`-Abhaengigkeit von `robust_best_score`

Dadurch wird nichts Wertvolles weggeworfen, aber `StudyRunner` hat wieder nur einen klaren Siegerpfad.

Wenn HP Robustness spaeter wieder gebraucht wird, dann bewusst als separater Modus oder separates Werkzeug, nicht als heimlicher zweiter Siegerpfad.

## Dashboard

Das Dashboard-Panel fuer Robustness bleibt wie in `design_icre.md` gebaut.

Sobald `evaluate_checkpoint_robustness(...)` Fortschritt meldet, rendert das Panel den Checkpoint-Robustness-Plot mit Kandidatenintervallen.

Der Panel-Titel bleibt progress-abhaengig, zum Beispiel:

```text
Checkpoint Robustness Evaluation - Candidate 2/3 - Eval 1/1
```

Die Integration ist dann wirklich integriert, weil der Runner die Fortschritte selbst erzeugt.

## Umsetzungsschritte

1. `StudyRunner` um `robust_eval_episodes: int = 20` erweitern.
2. `extra_seeds` aus `StudyRunner` entfernen.
3. Import in `study.py` von HP Robustness auf Checkpoint Robustness umstellen.
4. `StudyRunner.run(...)` nach `run_study(...)` `evaluate_checkpoint_robustness(...)` aufrufen lassen.
5. Beste Checkpoint-Robustness nach `robust_score` auswaehlen.
6. Trial-Params des Gewinner-Trials als neue Incumbent-Params uebernehmen, wenn der Score besser ist.
7. `_study_already_finished(...)` auf `checkpoint_robustness` umstellen.
8. Tests in `hpo/tests/test_study.py` auf den neuen Runner-Ablauf anpassen.
9. Notebook-Runner auf `robust_candidates=3`, `robust_eval_episodes=50` umstellen.
10. Separate Notebook-Zelle mit `evaluate_checkpoint_robustness(...)` entfernen.

## Nichtziele

- Keine neue Dashboard-API.
- Keine neue Reporter-Methode.
- Kein Umschalter zwischen HP Robustness und Checkpoint Robustness im Runner.
- Keine gewichtete Siegerformel.
- Keine per-Welt-Detailansicht im Overview.
- Keine granulare Progress-Meldung innerhalb eines Kandidaten.

## Offene Kleine Frage

Default fuer `robust_eval_episodes` soll erstmal `50` sein.

Das bedeutet bei SSL mit fuenf Welten `250` Landeversuche pro Checkpoint. Das ist ein brauchbarer Startkompromiss zwischen Zeitbedarf und Genauigkeit.

Fuer schnelle Dashboard-Smoke-Laeufe kann das Notebook niedriger gehen. Fuer finale Medaillenvergabe kann es spaeter auf `100` oder `200` erhoeht werden.
