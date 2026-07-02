## Pilot Preservation Ideas

- ==Early talent signal:== if the trailing mean-100 does not rise clearly within the first ~300-400 episodes, the trial is probably not a late Armstrong; validate this against Top- and Flop-Trials in the DB before turning it into an early stopping rule.
- ==Stop on Kelly-Bundy effect:== if `best_mean` was strong and the current trailing mean falls far below it, stop training and evaluate the preserved best checkpoint instead of the damaged final model.
- ==Roll back to local max:== when a promising pilot starts collapsing, restore the checkpoint from the previous local maximum and continue with gentler HPs, e.g. lower learning rate, epsilon bump, or slower update schedule.
- Low-epsilon guard: once `epsilon < 0.05`, continue only if new best means still appear; otherwise stop, reduce learning rate, or deliberately raise epsilon before further training.

**When:**
- 2026-06-30: initial pilot preservation ideas.
- 2026-07-02: added early talent signal.

## Video vom Training

**When:** 2026-06-30.

## Dashboard liest aus DB

**When:** 2026-06-30.


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

**When:** 2026-06-29.
