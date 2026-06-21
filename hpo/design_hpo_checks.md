## HPO-Diagnosechecks

Die Checks werden in einem separaten Modul `hpo/evaluation/checks.py` implementiert. Sie gehören zur Auswertung und nicht zu `study.py`.

```python
results = run_hpo_checks(runner.studies)
show_hpo_check_report(results)
```

Die Notebook-Ausgabe orientiert sich an Pytest:

```text
HPO plausibility checks

PASS  Training progress
WARN  Search-space boundary
PASS  Replay-memory utilization
PASS  Final exploration
WARN  World-score balance
WARN  Training-seed stability

Overall: 🟡 REVIEW — 3 warnings
```

Bei Warnungen folgen konkrete Messwerte:

```text
WARN Training progress
     Previous 50 episodes: 82.3
     Last 50 episodes: 134.1
     Training was still improving. Consider increasing num_episodes.
```

### Ampellogik

- 🟢 **PASS**: keine Auffälligkeiten
- 🟡 **REVIEW**: mindestens ein plausibler Hinweis
- 🔴 **FAIL**: Versuchsauswertung unbrauchbar, etwa weil notwendige Daten fehlen oder inkonsistent sind
- **SKIP**: Check ist nicht anwendbar und verschlechtert die Ampel nicht

### Checks

Die Checks werden iterativ ergänzt:

1. `training_progress`: letzte 50 gegen vorherige 50 Episoden
2. `final_exploration`: letztes Epsilon gegen `eps_end`
3. `world_balance`: schwächste Welt gegenüber dem Gesamtscore
4. `search_space_boundary`: Gewinner liegt am Rand
5. `replay_utilization`
6. `seed_stability`

Für `seed_stability` müssen die Ergebnisse der Robustness Evaluation dauerhaft in der Study gespeichert werden. Derzeit existieren sie nur während der Anzeige; dieser Check folgt deshalb später.

### Umsetzung

- neues Modul `hpo/evaluation/checks.py`
- neue Tests `hpo/tests/test_checks.py`
- eine kleine abschließende Notebook-Zelle
- `study.py` bleibt zunächst unverändert
