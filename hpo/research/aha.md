# HPO Aha

| Nr                                                  | Aha                                       | Topics                 |
| --------------------------------------------------- | ----------------------------------------- | ---------------------- |
| [[#A11 Preserve Good Pilots Immediately\|A11]]       | Preserve Good Pilots Immediately          | OTO, Checkpointing, LL |
| [[#A10 10D Gives The SSL A Popometer\|A10]]          | 10D Gives The SSL A Popometer             | SSL, RL                |
| [[#A9 Earth Is Learnable\|A9]]                        | Earth Is Learnable                        | SSL, RL                |
| [[#A8 Hard Worlds Need Their Own Flight Hours\|A8]]   | Hard Worlds Need Their Own Flight Hours   | SSL, Sampling          |
| [[#A7 Good HPs Are Not Enough\|A7]]                   | Good HPs Are Not Enough                   | RL, Checkpointing, LL  |
| [[#A6 Observation Mode Is Not Settled\|A6]]           | Observation Mode Is Not Settled           | SSL                    |
| [[#A5 Visualize Early\|A5]]                           | Visualize Early                           | OTO, LL, Dashboard     |
| [[#A4 Let Optuna Explore\|A4]]                        | Let Optuna Explore                        | OTO, Optuna, LL        |
| [[#A3 Gamma And Tau Shape Learning Dynamics\|A3]]     | Gamma And Tau Shape Learning Dynamics     | RL, Optuna             |
| [[#A2 Back Up Immediately\|A2]]                       | Back Up Immediately                       | OTO, Checkpointing, LL |
| [[#A1 Code Complexity Is Part Of The Experiment\|A1]] | Code Complexity Is Part Of The Experiment | OTO, LL                |

Topics: `RL` = Reinforcement Learning, `SSL` = SolarSystemLander, `OTO` = Optimize the Optimizer, `LL` = Lessons Learned.


## A11 Preserve Good Pilots Immediately

The 211 and 242 pilots were lost because good checkpoints were not preserved durably in time. After adding automatic best-eval checkpoint archiving to Drive, the 253 five-world 10D pilot survived a Colab runtime crash.

==Mental model: Do not trust the runtime. Bring the good pilot into the hangar.==

## A10 10D Gives The SSL A Popometer

10D gives the SSL a seat-of-the-pants sensor.

Ergebnis: 5-Welten-Gym-Score: 253

> [!quote] Codex:
> Vorher wusste der SSL eher: Wo bin ich? Wie schnell bin ich? Mit `dv_x/dt` und `dv_y/dt` spürt er zusätzlich: Was macht die Luft, Schwerkraft oder Steuerung gerade mit mir? Werde ich gerade beschleunigt oder abgebremst?
>
> Gerade bei fünf Welten ist das vermutlich Gold wert, weil die gleiche Geschwindigkeit je nach Welt etwas anderes bedeutet. Auf Mond, Erde und Venus fühlt sich "ich sinke mit `v_y`" dynamisch anders an. Die Beschleunigung sagt ihm etwas über das aktuelle Flugregime.

Das Popometer ist das anschauliche Bild; technisch bekommt der SSL eine Näherung für den aktuell wirkenden Kraftvektor.
## A9 Earth Is Learnable

The 9D SolarSystemLander can learn Earth. The Earth-only `s7_exploration` produced several `200+` candidates and showed that the earlier Earth weakness was not a physical impossibility, but a training setup problem.

==Mental model: Earth and Venus do not need pity; they need flight hours.==

## A8 Hard Worlds Need Their Own Flight Hours

Earth and Venus are hard because they need many own training episodes. In uniform five-world training, `1000` total episodes mean only about `200` per world, which is far below the Earth-only exposure that produced strong pilots.

## A7 Good HPs Are Not Enough

Good HPs are stochastic producers, not concrete models. Training seed and checkpoint choice matter enough that a strong HP set can still produce weak pilots, so concrete good checkpoints must be saved and evaluated.

==Mental model: HPs are producers, not models.==

**Model quality depends strongly on the training seed.** Earlier Elise studies already showed this: one seed reached about `167` mean score over five worlds, while robust re-evaluation of the same HPs fell to about `113` or `92`.

## A6 Observation Mode Is Not Settled

9D is the strongest current path because gravity helped on Earth without the questionable 11D wind/turbulence signals. But old 8D/11D comparisons used weaker HPs and shorter training, so 8D, 9D, and 11D still deserve a fair comparison later.

## A5 Visualize Early

The dashboard's colored training plot made the real problem visible: Earth and Venus were the hard worlds. Live plots are not decoration; they are diagnostic instruments.

==Mental model: The dashboard is our microscope.==

## A4 Let Optuna Explore

The Earth breakthrough came after letting Optuna search several HPs at once in a wider space. When the situation is unclear, narrowing too early can hide the path.

> Tagelang manuell gefummelt und kaum weitergekommen. Optuna in vielen Dimensionen suchen lassen und: Schwupps, gute HPs gefunden.
> -- angehender KI-Jedi-Schueler aus dem Maschinenraum

## A3 Gamma And Tau Shape Learning Dynamics

In RL, the third decimal place of parameters such as `gamma` and `tau` can matter a lot. Optuna is not just convenient here; it is the fine-adjustment tool that can find these sensitive settings without endless manual mouse-milking.

> [!quote] Codex:
> [...] `gamma` und `tau` sitzen beide im Herz der DQN-Lernmechanik, also bei TD-Target, TD-Error und Target-Netz-Update.
>
> `gamma`: Wie weit schaut der Lander in die Zukunft?
>
> Das TD-Target ist grob:
>
> ```
> target = reward + gamma * best_future_value
> ```
>
> [...]
>
> Wenn `gamma` von `0.99` auf `0.995` geht, wirkt das klein. Aber ==über viele Schritte potenziert sich das==:
>
> ```
> 0.99^100  ≈ 0.37
> 0.995^100 ≈ 0.61
> 0.99^200  ≈ 0.13
> 0.995^200 ≈ 0.37
> ```
>
> Bei `gamma=0.995` sind spätere Rewards also viel stärker “noch hörbar”. ==Für Lander wichtig: Die gute Landung oder der Crash kommen oft spät. Ein kleines `gamma` macht den Lander kurzsichtiger.==
>
> `tau`: Wie schnell folgt das Target-Netz?
>
> DQN hat zwei Netze:
>
> ```
> q_net        lernt ständig
> target_net   liefert stabilere Ziele
> ```
>
> Bei Soft Update:
>
> ```
> target_net = tau * q_net + (1 - tau) * target_net
> ```
>
> `tau=0.005` heißt: Target-Netz bewegt sich pro Update 0.5 Prozent Richtung neues Q-Netz.  
> `tau=0.002` heißt: nur 0.2 Prozent.
>
> Klingt winzig, aber das passiert sehr oft. Bei vielen Updates macht das enormen Unterschied: `tau=0.005` folgt deutlich schneller, `tau=0.002` ist träger und stabiler.
>
> Warum das bei uns wichtig ist:
>
> Der SSL fliegt mehrere Welten. Das Q-Net lernt ständig neue, teils widersprüchliche Erfahrungen. Wenn das Target-Netz zu schnell folgt, jagt das Lernziel hinter jedem Gewackel her. Dann kann Lernen instabil werden. Kleineres `tau` macht das Ziel ruhiger.
>
> Kurz:
>
> ```
> gamma = wie weit Zukunft zählt
> tau   = wie ruhig das Lernziel bleibt
> ```
>
> ==Und beide wirken nicht einmalig, sondern tausende Male in der TD-Lernschleife. Deshalb ist die dritte Nachkommastelle hier kein Schmuck, sondern Fahrwerksgeometrie.==

## A2 Back Up Immediately

The observed `211` and `242` pilots showed what was possible, but their concrete checkpoints were not saved in time. A good checkpoint only counts once it is preserved.

## A1 Code Complexity Is Part Of The Experiment

This follows from the [HPO research motto](README.md): correct HPO work improves the Gym score and keeps the code simple.

A better HPO algorithm is not only one that improves the Gym score. It must also stay understandable enough to keep the research loop alive and fast; otherwise software aging slows the experiment down.

LOC is a cheap first proxy for code complexity. It is imperfect, especially in Python formatting, but it gives high bang for the buck as an early warning signal when small ideas become large code.
