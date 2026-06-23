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


https://www.nasa.gov/the-apollo-program
