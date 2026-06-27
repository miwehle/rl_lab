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


# Vereinfachung der Objective

        # Idee:
        # - Folgendes in einem neuen hook evaluate_model machen.
        # - Danach den Hook q_net_for_evaluation entfernen.
        # - Also folgende Zeile statt "# { ... # }":
        # world_scores = hook.evaluate_model(ctx)
        # {
        ctx.q_net = ctx.training_result.q_net
        ctx.q_net = hooks.q_net_for_evaluation(ctx, ctx.trainer.device)

        ctx.world_scores = {
            name: evaluate_greedy_q_net(
                q_net=ctx.q_net,
                device=ctx.trainer.device,
                make_env=make_env,
                episodes=config.eval_episodes,
                max_steps=config.eval_max_steps,
                seed=config.eval_seed,
            )
            for name, make_env in config.environment_factory.evaluation_envs().items()
        }
        score = sum(ctx.world_scores.values()) / len(ctx.world_scores)
        ctx.score = score
        # }
