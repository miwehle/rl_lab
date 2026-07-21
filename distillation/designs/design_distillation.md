# Design: Elise Distillation

## Ziel

Ein separates Projekt `distillation` destilliert Elise-264-GSTP in einen kleineren DQN-Studenten. Startpunkt ist ein Student mit zwei Hidden Layers `64/64`, weil das die Parameterzahl grob um Faktor `3.6` reduziert (`18176 -> 4992` Gewichte) und damit als Kompression interessant ist.

## Projektgrenze

`distillation` ist kein HPO-Projekt und liegt daher auf Repo-Ebene:

```text
rl_lab/distillation/
```

Das Projekt darf `hpo` und `dqn` importieren. Das vermeidet eine zweite SolarSystemLander-Implementierung und hält HPO trotzdem klein.

```text
dqn -> DQN-Modellbasis, inklusive kleiner Studenten mit hidden_sizes=(64, 64)
hpo -> SolarSystemLander EnvFactory/Worlds/10D Observation, Checkpoint-Loading, ggf. Evaluation-Helfer
```

Nicht in V1:

```text
Optuna
StudyRunner
Dashboard
VectorTrainer
eigene SolarSystemLander-Env
```

## Notebook-Nutzung

Die Nutzung soll aus einem Notebook mit wenigen Zeilen möglich sein, analog zum Stil in `hpo/notebooks/solar_system_lander`. Der Normalfall kennt keine Infrastrukturdetails, keine Drive-Pfade und keine `InfraCfg`:

```python
from distillation import collect_teacher_dataset, train_student, evaluate_student, evaluate_teacher
from distillation import collect_teacher_dataset_parallel, plot_score_gaps, plot_score_quantiles, score_comparison_table

dataset = collect_teacher_dataset()
student = train_student(dataset)
student_summary = evaluate_student(student)
teacher_summary = evaluate_teacher()
plot_score_quantiles(teacher_summary, student_summary)
plot_score_gaps(teacher_summary, student_summary)
score_comparison_table(teacher_summary, student_summary, min_diff=25.0)
```

Für häufige Abläufe sind die Defaults ausreichend. Explizite Parameter sind nur dort nötig, wo experimentiert wird, z. B. `epsilon`, `seeds`, `student_hidden_sizes` oder `eval_episodes_per_world`. Teacher-Name, Dataset-Name und Run-Name haben Konventionen und sind nur optionale Overrides. Lange Notebook-Funktionen zeigen per Default eine Progressbar und können mit `progress=False` ruhig gestellt werden.

## Infrastruktur-Konventionen

Distillation bekommt eine eigene `InfraCfg`, analog zu `hpo.study.infra_cfg.InfraCfg` und `hpo.evaluation.video.InfraCfg`, aber nur für Infrastruktur. Fachliche Experimentparameter bleiben explizit.

```text
class InfraCfg:
    teacher_archive_dir: Path = /content/drive/MyDrive/rl_lab/hpo/best_checkpoints
    local_distillation_dir: Path = /content/rl_lab/distillation/runs
    drive_distillation_dir: Path = /content/drive/MyDrive/rl_lab/distillation

    def teacher_checkpoint_dir(teacher_name: str) -> Path:
        return teacher_archive_dir / teacher_name

    def teacher_checkpoint_path(teacher_name: str) -> Path:
        return teacher_checkpoint_dir(teacher_name) / "best_eval_checkpoint.pt"

    def dataset_dir() -> Path:
        return drive_distillation_dir / "datasets"

    def run_dir(run_name: str) -> Path:
        return drive_distillation_dir / "runs" / run_name

    def student_checkpoint_path(run_name: str) -> Path:
        return run_dir(run_name) / "student_checkpoint.pt"
```

Aufgaben:

```text
Google Drive in Colab mounten, falls verfügbar
lokale und Drive-Ordner anlegen
Teacher-Checkpoint über teacher_name finden
Checkpoint-Metadaten über hpo.checkpointing.checkpoint_metadata(path) laden
Dataset-Pfade ableiten
Run-/Student-Checkpoint-Pfade ableiten
Evaluation-Output-Pfade ableiten
Student-Checkpoints nach Drive sichern
```

Damit muss das Notebook keine Google-Drive- oder Checkpoint-Pfade enthalten:

```python
dataset = collect_teacher_dataset(epsilon=0.05, seeds=[7, 42, 1911, 4711], world_mix=WORLD_MIX)
student = train_student(dataset, hidden_sizes=(64, 64))
summary = evaluate_student(student, eval_episodes_per_world=100)
```

Die API nimmt `cfg` wie in `hpo.evaluation.video.record_video(...)` als optionalen Parameter mit Default. Im normalen Notebook wird `InfraCfg` nicht importiert und nicht erwähnt; es ist nur ein Override-Hebel für Sonderfälle. `teacher_name` und `run_name` sind ebenfalls optionale Overrides. Die High-Level-Funktionen übergeben diese Namen intern an `InfraCfg`; Notebook-Code ruft keine `InfraCfg`-Methoden auf.

```python
def collect_teacher_dataset(
    *,
    teacher_name: str = "solar_system_lander_10d_elise_stp",
    epsilon: float,
    seeds: Sequence[int],
    world_mix: Mapping[str, int],
    cfg: InfraCfg = InfraCfg(),
): ...

def train_student(
    dataset,
    *,
    hidden_sizes: tuple[int, int] = (64, 64),
    run_name: str | None = None,
    cfg: InfraCfg = InfraCfg(),
): ...
```

Konventionen gelten für Pfade und Artefaktablage. Fachliche Parameter wie `epsilon`, `seeds`, `hidden_sizes`, `world_mix` und `eval_episodes_per_world` sind keine InfraCfg-Defaults, sondern sichtbare Experimenthebel.

Die High-Level-Funktionen geben kleine Referenzobjekte zurück, keine nackten Pfade:

```python
@dataclass(frozen=True)
class DatasetRef:
    path: Path
    metadata: dict

@dataclass(frozen=True)
class StudentRef:
    checkpoint_path: Path
    metadata: dict
```

Konkreter Vorteil: Das Notebook bleibt kurz, aber Folgefunktionen bekommen Pfad und Metadaten zusammen. `train_student(dataset)` muss die Dataset-JSON nicht erneut suchen, `evaluate_student(student)` kennt Architektur und Checkpoint-Pfad, und bei Bedarf kann das Notebook trotzdem `dataset.path` oder `student.metadata` anzeigen.

## Verfahren

V1 ist offline Q-value distillation:

1. Elise-264-GSTP als Teacher laden.
2. Über Worlds und Seeds Rollouts sammeln.
3. Pro Step speichern:

```text
observation      [10]
teacher_q_values [4]
teacher_action
world
seed
step
scenario
```

`teacher_q_values` werden direkt beim Sammeln gespeichert, nicht erst beim Training neu berechnet. Dadurch ist das Dataset stabil, mehrere Students können dasselbe Dataset verwenden, Training/Validation brauchen den Teacher nicht nochmal, und Ergebnisse bleiben leichter reproduzierbar.

4. Student `64/64` trainieren:

```text
loss = MSE(student_q(observation), teacher_q_values)
```

5. Student greedy im Gym evaluieren und mit Teacher vergleichen.

## Dataset

Die wichtigste freie Variable ist die Zustandsverteilung. V1 sammelt sequentiell, nicht vektorisiert:

```text
greedy teacher rollouts
epsilon teacher rollouts, z. B. epsilon = 0.05
bekannte harte Seeds, z. B. Earth/Venus seed=10014
optional Starkwetter/Down-Kick-Szenarien
```

`seeds` ist der primäre Hebel für die Bedingungsverteilung. Der Seed bestimmt unter anderem Gym-Startbedingungen, Terrain/Reset-Zufall, initialen Kick sowie Wind- und Turbulenzstärke im HPO-`EnvWrapper`. Damit kann das Notebook den Schwierigkeitsgrad bewusst wählen, z. B.:

```python
seeds = list(range(1000, 2000)) + [7, 42, 1911, 10014]
dataset = collect_teacher_dataset(epsilon=0.05, seeds=seeds, world_mix=WORLD_MIX)
```

Wie im HPO-Trainingsnotebook kann `world_mix` schwere Welten stärker gewichten, z. B. `{World.MERCURY: 1, World.VENUS: 4, World.EARTH: 4, World.MOON: 1, World.MARS: 1}`.

`epsilon` ist ein Dataset-Parameter, aber noch kein Grund für Optuna. Erst kleine manuelle Sweeps testen, z. B. `0.00`, `0.05`, `0.10`.

## Training

`train_student(...)` trainiert einen kleinen `dqn.model.DQN` per supervised learning auf gespeicherte Teacher-Q-Werte:

```python
student = train_student(dataset, hidden_sizes=(64, 64))
```

V1-Student-Modell:

```text
input 10
hidden 64
hidden 64
output 4 Q-values
ReLU zwischen den linearen Layers
linearer Output-Layer
```

Der Student nutzt dieselbe `DQN`-Klasse wie der Teacher, nur mit `hidden_sizes=(64, 64)` statt Elises größerem symmetrischem Netz. Dadurch gibt es keinen duplizierten Modellcode in `distillation`.

Loss:

```text
MSE(student_q(observation), teacher_q_values)
```

Das Dataset wird in Train/Validation gesplittet. Gespeichert werden:

```text
student_checkpoint.pt
student_checkpoint.json
training_summary.json
```

Die Student-Metadaten enthalten mindestens:

```text
student_hidden_sizes
teacher_name
teacher_checkpoint_path
dataset_path
train_loss
val_loss
val_argmax_agreement
training parameters
```

`train_student(...)` gibt einen `StudentRef` zurück. Der Checkpoint wird nach Drive geschrieben; das Notebook muss keinen Zielpfad kennen.

## Qualitätsmessung

Billige Imitationsmetriken:

```text
validation MSE(student_q, teacher_q)
argmax agreement
```

Entscheidend ist Gym-Qualität:

```text
mean score
world_scores
min/q05/median/q95
score gap zu Elise
schwächste Welt
```

Für V1 reicht: mean score plus world_scores separat anschauen. Keine Metrikmagie.

## Evaluation

`evaluate_student(...)` lädt den Student-Checkpoint, baut dieselben HPO-SolarSystemLander-Evaluation-Envs und führt greedy Rollouts aus:

```python
summary = evaluate_student(student, eval_episodes_per_world=100)
```

`evaluate_teacher(...)` nutzt dieselbe Evaluation für den Teacher-Checkpoint und gibt dieselbe Summary-Struktur zurück. Dadurch können Teacher und Student direkt verglichen werden.

Die Funktion speichert `evaluation_summary.json` im Run-Ordner und gibt dieselbe Zusammenfassung als Dict zurück:

```text
checkpoint_path
eval_episodes_per_world
episodes
mean
median
min
q05
q25
q75
q95
max
world_scores
teacher_name
dataset_path
student_hidden_sizes
```

V1 hält es schlicht: Teacher und Student greedy evaluieren, Scores pro Welt anschauen, fertig.

Für den Notebook-Vergleich gibt es zwei kompakte Plot-Helfer:

```text
plot_score_quantiles(teacher_summary, student_summary)
plot_score_gaps(teacher_summary, student_summary)
score_comparison_table(teacher_summary, student_summary, min_diff=25.0)
```

Beide zeigen die einzelnen Welten plus eine Zeile bzw. Spalte über alle Welten. Der Quantile-Plot nutzt das Robustness-Muster: q05..q95, q25..q75, Median und Mean.

Die Tabelle paart Teacher und Student nach `(world, seed)`, zeigt `teacher_score`, `student_score` und `teacher_minus_student`, filtert kleine Differenzen über `min_diff` heraus und kann mit `ascending=True` die Fälle zeigen, in denen der Student besser ist.

## Tests

Automatisierte Tests bleiben in V1 klein und schnell:

```text
InfraCfg-Pfadkonventionen ohne Drive-Mount
DQN-Shape für Student-Größe: [batch, 10] -> [batch, 4]
Dataset speichern/laden mit kleinem künstlichem Dataset
train_student smoke test mit synthetischem Dataset und wenigen Epochs
evaluate_teacher/evaluate_student smoke tests mit Fake-Env
Plot-Helfer mit synthetischen Summaries
Score-Vergleichstabelle mit synthetischen Summaries
```

Nicht in V1 automatisiert:

```text
lange Gym-Rollouts
echte Elise-Qualität
Colab/Google-Drive-Mount
Optuna
```

Gym-heavy Dataset-Sammlung und robuste Student-Evaluation bleiben zunächst Notebook-/Integration-Checks. Tests prüfen die Nähte und Artefaktkonventionen, nicht die fliegerische Qualität.

Falls Dataset-Sammlung zum Engpass wird, bleibt `collect_teacher_dataset(...)` die einfache Referenzversion. Zusätzlich kann das Notebook auf `collect_teacher_dataset_parallel(..., num_envs=16)` umschalten; diese Variante lebt in einem eigenen Modul und nutzt `AsyncVectorEnv` batchweise, damit die einfache Version nicht verschachtelt wird.

## Artefakte

Distillation-Artefakte gehören unter `drive_distillation_dir`, nicht ins HPO-Teacher-Archiv. Ein Run speichert mindestens:

```text
dataset metadata
student checkpoint
student training summary
student evaluation summary
```

Konventionelle Ablage:

```text
/content/drive/MyDrive/rl_lab/distillation/
  datasets/
  runs/
    <run_name>/
      student_checkpoint.pt
      student_checkpoint.json
      training_summary.json
      evaluation_summary.json
```

Der lokale `local_distillation_dir` ist Arbeitsbereich/Cache. Der wertvolle Student-Checkpoint wird immer über `cfg.student_checkpoint_path(run_name)` nach Drive geschrieben. Wenn mehrere Studenten oder Trainingsvarianten entstehen, bekommen sie unterschiedliche `run_name`s. Im Normalfall erzeugt `train_student(...)` den `run_name` aus Teacher, Student-Architektur, Dataset-Kennung und Zeitstempel; nur bewusst vergleichende Experimente setzen ihn explizit.

Teacher und Student folgen beide der Konvention aus `hpo.checkpointing`: Gewichte in `.pt`, Metadaten als `.json` direkt daneben. `InfraCfg` liefert nur den `.pt`-Pfad; der Metadata-Pfad wird per `checkpoint_metadata_path(path)` abgeleitet. Student-Metadaten enthalten mindestens Architektur, Teacher-Referenz, Dataset-Referenz, Trainingsparameter und Evaluationsergebnis.

## Laufzeit

Lokaler Smoke Test ist sinnvoll. Echtes Dataset und Training laufen wahrscheinlich gut auf Colab L4. Dataset-Sammlung und Evaluation sind eher Gym/CPU-lastig; Student-Training und Teacher-Q-Berechnung profitieren etwas von GPU, sind aber klein.

VectorEnv ist optional für später, falls Dataset-Sammlung nervt. VectorTrainer passt nicht zur reinen Distillation.

## KISS-Regeln

```text
Use HPO. Don't become HPO.
Dataset statt Replay Buffer.
Supervised Training statt Target Network.
Erst 64/64 beweisen, dann kleinere Studenten oder Fine-Tuning.
```
