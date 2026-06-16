## Basic Optuna workflow

Optuna is a Python package for automated hyperparameter (HP) optimization.

Optuna is used like this:
- Define a function `objective(trial)` 
- Create a `study` and call `optimize(objective, n_trials)`
- Evaluate the trials, i.e. get the best HPs

### As UML diagram

After defining the `objective` function, use Optuna like this:

```mermaid
%%{init: {"sequence": {"mirrorActors": false}}}%%
sequenceDiagram
    actor user
    create participant study as :Study

    user->>study: create_study()
    user->>study: optimize(objective, n_trials=20)
    user->>study: get best_params
```

### As Python code

```python
import optuna

def objective(trial):
    # ... (It's the user's job to define this function.)
    return score

# Now the user can push the Optuna button, so to speak, to get the best HP values:
study = optuna.create_study()
study.optimize(objective, n_trials=20)

best_params = study.best_params
print(best_params)
```

***What does `objective` do?***

`objective(trial)`:
- call `trial.suggest_*` to get HP suggestions
- train the model with these HP values
- score the model
- return the `score`

Example: `trial.suggest_float("lr", 1e-4, 1e-3)`

Under the hood: In order to suggest an HP value, trial
- asks study.sampler for a value
- stores the suggested value in the trial/study storage
- returns the value to objective


***What does `optimize` do?***

`optimize(objective, n_trials)`: For `n_trials` times:
- create a `trial`
- call `objective(trial)`
- suggest HP values when `objective` calls `trial.suggest_*`  # remove?
- store `(hp_values, score)`

### Essence

Define the `objective` -> create a `study` -> run `study.optimize(...)` -> evaluate the trials.

### Links

[Simple example](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html) in the Optuna tutorial.


## Lunar Lander HPO design

### Gegeben

#### Training

> Mean score verläuft manchmal V-förmig über den Episoden eines Trainings.

Training = Ausführung von trainer.train(...) (dauert ca. 3 min).

> Während des Trainings ändert sich epsilon.

epsilon = Explorationsrate

#### Optuna

> objective(trial):
> - ruft trainer.train(...) 1x auf  
> - speichert in trial Infos: Bestes mean score Fenster u. a.

> Am Ende einer Studie: Mit study.best_trial kommt man an die Infos vom besten Trial. 

#### Hyperparameter

Das sind alle Hyperparameter im Subprojekt dqn:
- learning_rate
- batch_size
- eps_end
- eps_decay
- gamma
- tau
- learning_starts
- optimize_every
- replay_memory_capacity
- double_dqn

### Ansatz

#### Studien

Studie 0: Baseline festlegen  
Kein echter HPO-Lauf, eher Referenz:

Dann:
> Mehrere Studien mit je ca. 20 bis 40 Trials
> - Zeitbedarf: Studie mit 40 Trials dauert 2 h (40 * 3 min)
> - Optuna findet in jeder Studie mittels TPE die besten Werte für ausgewählt HPs
> - Welche HPs in welchen Bereichen optimiert werden: Das wird ***SearchSpace*** festgelegt (siehe HPO-Notebook).

### Definition der Studien

1. Update-Ökonomie
   learning_rate + batch_size + optimize_every + learning_starts

2. Exploration
   eps_decay + eps_end

3. Replay-Kapazität nur kurz prüfen
   replay_memory_capacity

4. Kleine gemeinsame Feinsuche
   enge Bereiche um die Gewinner

5. Top-Kandidaten mit mehreren Seeds bestätigen

#### Suchraum von Studie 1