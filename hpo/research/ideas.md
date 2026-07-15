# HPO Ideas
## Pilot Preservation

- ==Early talent signal:== if the trailing mean-100 does not rise clearly within the first ~300-400 episodes, the trial is probably not a late Armstrong; validate this against Top- and Flop-Trials in the DB before turning it into an early stopping rule.
	- ==Aber: Manche der besten Modelle fingen nicht auffällig gut an.== Erst in einer späteren Lernwelle wurden sie gut.
- ==Stop on Kelly-Bundy effect:== if `best_mean` was strong and the current trailing mean falls far below it, stop training and evaluate the preserved best checkpoint instead of the damaged final model.
	- ==Aber: Lernen passiert öfter in Wellen.== Manchmal sieht es erst aus wie ein Kelly-Bundy-Effekt. Aber es ist Anlauf nehmen und dann neuen Highscore aufstellen.
- ==Roll back to local max:== when a promising pilot starts collapsing, restore the checkpoint from the previous local maximum and continue with gentler HPs, e.g. lower learning rate, epsilon bump, or slower update schedule.
	- ==Aber: Kein voreiliges Rollback== (wegen *Lernen in Wellen*).
	- ==Wohl besser:== Versuchen, ==Highscorer== mit modifizierten HPs oder anderen Environment-Häufigkeiten ==noch weiter zu verbessern==.
- Low-epsilon guard: once `epsilon < 0.05`, continue only if new best means still appear; otherwise stop, reduce learning rate, or deliberately raise epsilon before further training.

## Video vom Training

- Videos sollten optional ==weltabhaengige, realistischere Farben== bekommen, z. B. blauer Himmel und gruener Boden fuer Earth, gelbliche Venus, roetlicher Mars, grauer Moon/Mercury. Das macht Videos intuitiver lesbar und hilft, Landungen pro Welt schneller einzuordnen.

## Dashboard
### Daten aus DB lesen

- Damit man die Dashboard-Daten nicht durch die halbe Implementierung durchreichen muss. ==Das könnte die Impl. deutlich vereinfachen.==
	- Wobei: Das verwendete ==Hook-Pattern== hilft wohl schon.
### Robustness Evaluation
- in RL oder Q-Learning zumindest aber ==bei SSL: Checkpoint Robustness ist wichtiger als HP Robustness==

## Begriffe aus dem Apollo-Programm

Apollo-Programm
└── Saturn-V-Entwicklungsprogramm
    ├── Entwicklungsmodelle
    ├── Testkampagnen
    ├── Qualifikationstests
    └── Flugerprobung

-->

SSL-Entwicklungsprogramm
├── Entwicklungsmodelle       → gespeicherte Checkpoints
├── Testkampagnen             → HPO- und Trainingsfolgen
├── Qualifikationstests       → feste Evaluation über fünf Welten
└── Flugerprobung             → robuste Prüfung mit neuen Seeds


Raumfahrt ist organisierte Paranoia mit Checklisten.

Die Begriffe überschneiden sich, prüfen aber unterschiedliche Reifegrade:

- **Testkampagnen:** Viele systematische Versuche, um Fehler zu finden und das Design zu verbessern.
- **Qualifikationstests:** Nachweis, dass das konkrete Design die festgelegten Anforderungen erfüllt.
- **Flugerprobung:** Prüfung unter möglichst realen Einsatzbedingungen.

Bei uns:

- HPO erzeugt und verbessert Entwicklungsmodelle.
- Die Qualifikation prüft `Score > 200` auf jeder Welt unter festen Bedingungen.
- Die Flugerprobung prüft das qualifizierte Modell mit neuen Seeds und Wetterlagen.

Damit wird nichts doppelt geprüft: erst entwickeln, dann Anforderung nachweisen, dann der Realität eine Gelegenheit geben, uns zu demütigen. Murphy bekommt keinen Sitzplatz, reist aber erfahrungsgemäß trotzdem mit.


nasa.gov/the-apollo-program

## DQN Trainer Playground

Ein einfaches Notebook unter `hpo/notebooks/lunar_lander/playground.ipynb` koennte helfen, mit dem einfachen `dqn.training.Trainer` und echten Komponenten vertraut zu werden. Es soll kein Test-Runner sein, sondern ein kleiner Spielplatz mit Prints, Tabellen und Plots.

```text
# DQN Trainer Playground

## Setup
## Make Environment
## Random Episode
## Create Trainer
## Train 100 Episodes
## Inspect Learning
## Continue Training
## Greedy Episode / Animation
## Things To Try
```

## Tool-Assisted RL Agent Development

Das HPO-Dashboard koennte zu einem allgemeineren RL-Entwicklungstool wachsen: HPO, Checkpoint-Erhaltung, robuste Evaluation, Videos, interaktives Spielen und tool-assistierte Failure-Analyse werden zu einem kurzen Entwicklungsloop.

Case Study: SolarSystemLander / Elise.

Zielbild: Mit wenigen manuellen Entscheidungen und guter Dashboard-Auswertung purzeln spezialisierte Agenten oder Werkzeuge heraus, z. B. eine robuste Multi-World-Elise, Reward-Shaping-Varianten oder eine Safety-Elise als Retterin/Shield.

Demo-Idee: ein auf Knopfdruck installierbares Multi-Planet-Lander-Spiel, in dem Menschen selbst landen und Elise als Autopilot/Retterin zuschalten koennen. Das ist vor allem Cool-Faktor und Intuitionsanker, aber es nutzt dieselbe Physik, dieselben Welten und dieselben Seeds wie die harte Evaluation.

Harte Evidenz darf nicht zu kurz kommen: Score-Verteilungen ueber Welten und Seeds, Robustness-Tabellen, HPO-Verlaeufe, Checkpoint-Vergleiche, Failure-Mode-Metriken wie `landed-but-awake`, sowie Diagramme fuer Recoverability/Safety-Shielding.

Paper-Winkel: Tool-assisted RL agent development ueber Environment-Familien hinweg. Elise waere die anschauliche Case Study, nicht die Grenze des Tools.

## Proper-Acceleration Popometer

The current 10D SolarSystemLander observation appends clipped velocity deltas (`dv_x`, `dv_y`). This measures net acceleration. A pilot-like Popometer might instead expose felt force per mass, roughly `dv/dt - gravity`: hovering has `dv/dt ~= 0`, but it is not force-free.

Idea: compare the current 10D acceleration observation with a proper-acceleration variant: `8D + proper_accel_x + proper_accel_y`.

Reason: A lander pilot would also feel gravitational force, not just net velocity change. This may help one shared agent behave more consistently across worlds with different gravity.

Status: loose idea; turn into a hypothesis only after defining scaling, clipping, and a fair comparison.

## Coupled Batch Size And Optimize-Every Search

Idea: test a small coupled Optuna axis for GPU-efficient DQN updates:

```python
trial.suggest_categorical("batch_opt_pair", [
    (512, 2),
    (1024, 4),
    (2048, 8),
    (4096, 16),
])
```

Rationale: larger batches use the GPU much more efficiently per replay sample, while larger `optimize_every` reduces update frequency and wall time. Coupling both keeps the rough replay-samples-per-env-step budget comparable, but changes the learning dynamics: fewer, larger, more strongly averaged gradient steps.

Risk: the mean loss over large batches may remove useful stochasticity and make DQN learning slower or worse, even if the wall-clock throughput improves.

KISS test: start with this coupled axis only, not a full Cartesian product. If it looks promising, later decide whether to decouple `batch_size` and `optimize_every`.

Related observations: [[observations#O16 AsyncVectorEnv Speeds Up SSL Env Stepping On Colab|O16]] and [[observations#O17 AsyncVectorEnv Gives Small VectorTrainer Speedup On L4|O17]]. The update microbenchmark showed that batch `4096` costs only about `1.28x` as much per update as batch `512`, while processing `8x` as many replay samples.

## Pack Learning: Shared Replay With Model-Specific Prioritized Replay

Working name: Pack Learning (Gruppenlernen). Short description: shared experience, different policies, model-specific surprise.

Idea: build toward a trainer that gets more learning out of expensive environment experience by combining shared replay, multiple DQN learners, and model-specific prioritized replay.

Two-stage path:

1. Add prioritized experience replay (PER) for the current single-model `VectorTrainer`.
2. Add multi-model training where several DQNs share the same transition storage but keep model-specific priority views.

Core shape:

```text
shared transitions:
  states, actions, rewards, next_states, done

model-specific priorities:
  priorities[model_id, transition_id]
```

Each model samples according to its own surprise:

```text
DQN A samples transitions surprising for A.
DQN B samples transitions surprising for B.
DQN C samples transitions surprising for C.
```

Mental model: experience is shared; surprise is model-specific.

Human analogy: much progress comes from group learning. One person learned how to make fire; others did not need to rediscover it from scratch. Fleming found penicillin by accident; medicine advanced because others learned from, refined, and transferred the discovery. Likewise, exploration does not have to stay private: if one model creates a surprising or useful experience, all models can learn from it according to their own gaps.

Important distinction: Pack Learning shares experiences, not answers. A model does not directly copy another model's policy. It learns from transitions created by others and evaluates them through its own Q-values, TD targets, and TD errors. Good examples can still help through reward and TD target, but not as directly as demonstration or imitation learning.

Possible later addition: success replay or positive priority. Surprise-based PER emphasizes high TD-error transitions, often including mistakes and failures. Successful rare sequences, good recoveries, and high-return landing maneuvers may deserve their own replay priority so they are not washed out by large mean-loss batches.

Useful priority may be the intersection of example value and model-specific surprise. If an `(observation, action)` pair led to high reward or high return for one model and another model is surprised by it, that transition is especially valuable for the surprised model. Positive TD-error means "this was better than expected" and can support learning from good examples; negative TD-error means "this was worse than expected" and can support learning from bad examples.

A later priority function could therefore separate or combine:

```text
positive_surprise = max(td_target - q_value, 0)
negative_surprise = max(q_value - td_target, 0)
success_signal    = high reward, high return, landing, recovery
failure_signal    = crash, bad return, unsafe state, lost landing reward
```

Then each model can learn from good and bad examples produced by others, without directly imitating their policy.

Potential benefit: simulation/experience generation is expensive, while additional learning from existing experience can be GPU-efficient if batched well. Multiple models could explore HP variants such as `lr`, `tau`, `gamma`, and `epsilon` while learning from a shared stream of experience. Model-specific PER may also protect rare high-TD-error transitions from being washed out by large mean-loss batches.

Exploration angle: shared replay across different models may reduce the need for high epsilon, because policy diversity creates structured exploration. What is routine behavior for one model can be novel and informative for another. This should not be treated as "epsilon becomes unnecessary" yet; epsilon remains a simple guard against early policy collapse and unreachable states. But a flight school with diverse pilots may need less random stumbling than one isolated pilot.

Important design point: avoid global PER as the first serious design, because global `max` or `mean` priority would flatten the most interesting part. The same transition may be boring for one model and highly informative for another.

Key open design freedom: who flies the next missions that create new experience for the shared replay buffer?

Possible behavior-policy variants:

- Round-robin: each model controls some env slots or episodes.
- Best-so-far pilot: the strongest current model generates most experience.
- Scout pilot: one deliberately explorative model generates varied experience.
- Mixed policy: combine best-so-far, scouts, and weaker learners so the replay library does not collapse into one narrow behavior distribution.

This matters because the replay buffer is shared, but the data distribution is still produced by concrete behavior policies. The class gets a shared library, but someone still chooses which field trips happen.

Likely complexity: PER itself is moderate and well-contained. Multi-model orchestration is the larger change: model-specific HPs, behavior policy for generating actions, fair comparison between learners, and later integration with HPO reporting.

KISS path: prototype PER first as a deep replay module, with an API that can later accept `model_id=0`. Then build the multi-model layer on top only if PER proves useful.
