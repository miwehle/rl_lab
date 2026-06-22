# Design Elise

Der Kern bleibt:

```text
Search Space → Optuna → robuste Auswahl → Incumbent → Dashboard
```

Alles, was nicht direkt hilft, einen hohen mittleren Gym-Score über fünf Welten zu erreichen, bleibt zunächst außen vor.

## Lotus-Elise-Version

- Das Objective ist ausschließlich der mittlere Gym-Score über die fünf Welten.
- Es gibt einen einzigen Begriff: `score`.
- `trial.value` ist überall die Wahrheit.
- Es gibt keine Quality/Effort-Kombination.
- Es gibt keine Aufwands-Baseline.
- Es gibt keine `baseline_env_steps` oder `baseline_processed_samples`.
- Es gibt keinen `training_effort`.
- Es gibt keinen `robust_best_gym_score`, weil `robust_best_score` bereits der Gym-Score ist.
- `robust=False` und die künstliche `s0`-Studie entfallen.

Die Baseline enthält nur:

```python
@dataclass(frozen=True)
class Baseline:
    params: dict[str, Any]
    score: float
```

Der `StudyRunner` startet mit dieser Baseline, führt reguläre Studien `s1`, `s2`, … aus und speichert nach jeder Studie den vollständigen Incumbent:

```text
incumbent_params
incumbent_score
```

Das Dashboard zeigt Trials, robuste Nachprüfung und den aktuellen Incumbent anhand genau dieses einen Scores.

Später können Aufwand, Multi-Objective-Scoring oder zusätzliche Metriken außen herum ergänzt werden. Der schnelle, leichte Kern muss davon nichts wissen.

Das Ziel ist nicht bloß weniger Code, sondern ein klareres Modell: Optuna optimiert genau das, was im Dashboard steht und was der `StudyRunner` als Gewinner übernimmt.
