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
