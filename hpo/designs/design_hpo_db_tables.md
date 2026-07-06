# HPO DB Tables

## Entscheidung

HPO-spezifische Metadaten bekommen eigene HPO-Tabellen in derselben SQLite-DB wie die Optuna-Study.

Optuna-Tabellen bleiben Optuna-Land. HPO-Tabellen sind unser Bereich fuer Daten, die nicht sauber in das Optuna-Schema gehoeren.

Die erste gesetzte Tabelle heisst:

```sql
hpo_study_metadata
```

Sie speichert Study-bezogene HPO-Metadaten.

Erstes KISS-Schema:

```sql
CREATE TABLE IF NOT EXISTS hpo_study_metadata (
    study_name TEXT PRIMARY KEY,
    runtime_provider TEXT NOT NULL,
    runtime_metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

`runtime_provider` beschreibt grob, wo die Study lief, zum Beispiel `colab`, `lambda`, `runpod`, `home` oder `work`.

## Runtime Metadata JSON

`runtime_metadata_json` enthaelt zuerst eine gemeinsame Schnittmenge automatisch ermittelbarer Felder und danach optional provider-spezifische Details.

Skizze:

```json
{
  "python_version": "...",
  "platform": "...",
  "cpu": "...",
  "torch_version": "...",
  "optuna_version": "...",
  "device": "cuda",
  "accelerator_backend": "cuda",
  "accelerator_name": "NVIDIA L4",
  "accelerator_count": 1,
  "git_commit": "...",
  "git_dirty": false,
  "colab": {
    "hardware_accelerator": "L4 GPU"
  }
}
```

Provider-spezifische Klammern wie `colab`, `lambda` oder `runpod` sind optional und sollen nur Felder enthalten, die fuer diesen Anbieter wirklich sinnvoll sind.

`runtime_metadata_json` soll kompakt bleiben. Die gemeinsamen Felder sollen vor allem helfen, Laufzeit- und Ergebnisunterschiede spaeter einzuordnen: Python, Plattform, CPU, Torch, Optuna, Accelerator und Git-Stand. `git_dirty` ist bewusst enthalten, weil Notebook-/Colab-Laeufe sonst leicht wie ein sauberer Commit aussehen, obwohl lokale Aenderungen aktiv waren.

Nicht geplant sind erstmal umfangreiche Inventarlisten wie kompletter Package-Freeze, RAM-Details, Hostname oder detaillierte CUDA-Treiberinformationen. Solche Felder kommen nur hinzu, wenn sie fuer eine konkrete Auswertung gebraucht werden.
