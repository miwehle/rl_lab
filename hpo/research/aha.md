# HPO Aha

| Nr                                                  | Aha                                       | Topics                 |
| --------------------------------------------------- | ----------------------------------------- | ---------------------- |
| [[#A16 Elise Has A Fight-Or-Relax Neuron\|A16]]     | Elise Has A Fight-Or-Relax Neuron         | SSL, RL, NN Viz        |
| [[#A15 Artificial Armstrong Is A Narrow Domain Criterion\|A15]] | Artificial Armstrong Is A Narrow Domain Criterion | SSL, RL, Evaluation |
| [[#A14 Efficient Pilots Need Dynamic Safety Margin\|A14]] | Efficient Pilots Need Dynamic Safety Margin | SSL, RL, Video      |
| [[#A13 Fine-Tune Strong Pilots Into A Model Line\|A13]] | Fine-Tune Strong Pilots Into A Model Line | SSL, RL, HP, Checkpointing |
| [[#A12 How To Build A Small Good Five-World Lander\|A12]] | How To Build A Small Good Five-World Lander | SSL, RL, HP, OTO       |
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

Topics: `RL` = Reinforcement Learning, `SSL` = SolarSystemLander, `OTO` = Optimize the Optimizer, `LL` = Lessons Learned, `HP` = Hyperparameters.

## ~~A16 Elise Has A Fight-Or-Relax Neuron~~

**Update:** ==This Aha is likely wrong in its original form.== Later activation analysis in [[observations#O18 H1-80 Is Strongly Wired But Inactive In Greedy Flights|O18]] showed that `H1-80` stays inactive in tested Elise-264-GSTP greedy rollouts, including hard Earth/Venus no-noop cases. The better current hypothesis is [[hypotheses#H10 H1-80 Ist Ein Alter Low-G-Overcontrol-Guard|H10]]: `H1-80` may be a dormant or rarely used low-g overcontrol guard rather than the active Fight-or-Relax neuron.

Neuron `H1-80` in Elise-264-GSTP is a surprisingly central hidden feature. In the `layer2.weight` matrix, it is the strongest hidden-1 source by a large margin: `sum(abs outgoing weights)) ~= 293.7`, versus about `156.6` for the next strongest H1 neuron. It is the strongest incoming H1 connection for `83/128` hidden-2 neurons.

Its input weights also make semantic sense. Sorted by absolute value, the strongest inputs are `dv_y = +1.176`, `left leg = -0.596`, `vy = +0.592`, `x = +0.462`, `dv_x = +0.450`, and `right leg = -0.304`. This suggests a coarse but useful signal: touchdown damps it, while vertical dynamics, vertical velocity, horizontal target offset, and horizontal disturbance increase it.

==Mental model: H1-80 is Elise's Fight-or-Relax neuron: are we settled enough to relax, or is control urgency high enough that the pilot must fight?==

This refines the popometer idea. The added `dv_x/dt` and `dv_y/dt` observations did not merely improve aggregate score; the trained network appears to have built a central internal control-urgency feature around them, especially around vertical acceleration.

## A15 Artificial Armstrong Is A Narrow Domain Criterion

`Artificial Armstrong` is not a claim that Elise has Neil Armstrong's broad human skill, judgment, courage, or creativity. It is a narrow and respectful analogy for one specific aspect: landing a small simulated lander across Moon, Mercury, Mars, Earth, and Venus under the current SolarSystemLander physics.

Elise-264-GSTP is more than a high score. It is a small, robust, fuel-efficient five-world pilot that usually lands in controlled fashion, and whose remaining worst failures increasingly look close to the physical/action-space boundary rather than ordinary pilot mistakes.

In this narrow sense, Elise reaches a surprisingly high fraction of an Armstrong-like landing criterion: broad cross-world competence, efficient control, robustness under disturbance, and failures that can be explained by control authority limits.

==Mental model: Artificial Armstrong is a domain-specific benchmark, not a human equivalence claim.==

## A14 Efficient Pilots Need Dynamic Safety Margin

Elise-264-GSTP's worst crashes show that a high-scoring pilot can still have a characteristic failure style: efficient, fast, fuel-saving descents with little reserve. This is often the right tactic against wind because it reduces time airborne, but it becomes fragile when wind or turbulence reverses late.

The crash-analysis videos made this visible. In the worst crash, turbulence changed from about `+2.1` to `-2.2 rad/s^2`, or roughly `+120` to `-126 deg/s^2`, and wind also reversed near touchdown. Elise had already committed to a fast descent, so the disturbance switch hit during the phase where she had least time and height left to re-stabilize. A turbulence reversal of about `+/-120 deg/s^2` near touchdown is a serious attitude-control problem even on an Armstrong scale.

==Mental model: A strong pilot does not only need skill; it needs adaptive safety margin when the air stops being a partner.==

For future tuning, this argues against a blunt "always descend slower" rule. The more promising direction is conditional caution: be more conservative near the ground when sink rate, rotation, side drift, or changing disturbance cues indicate that the efficient approach has become brittle.

The `seed=10014` Earth/Venus failures sharpen this further: some landings may be unrecoverable even on an Armstrong scale within the current discrete action space. Elise used no no-op steps, spent roughly half the episode on the main engine and roughly half on side-thrust attitude control. Attitude control is not optional here; when the lander rotates hard, main thrust cannot reliably oppose gravity. If turbulence consumes nearly half the action budget just to keep the nozzle useful, high-g worlds may leave too little thrust authority to arrest descent at all.


## A13 Fine-Tune Strong Pilots Into A Model Line

Elise-264-GSTP shows that a good pilot does not have to be the end of the search; it can become the parent of a model line. The 264er Elise starts from the preserved 253er 10D checkpoint and fine-tunes that concrete pilot with a small training-only Ground-Side-Thrust-Penalty.

The new checkpoint's source score reached `266.1`, but the more meaningful identity is the robustness mean around `264`, so `Elise-264-GSTP` is the better name. The old 253er was strong, but that number was a lighter study score; later broader checkpoint evaluation measured it lower. The 264er is therefore not just a higher peak, but a more robust concrete pilot.

GSTP was a record breaker already in the smoke test. The first small fine-tuning sweep found the new top pilot in Optuna trial `4` (zero-based numbering), which is strong evidence that the intervention hit a real weakness rather than merely adding search budget.

Success HPs for trial `4`: `learning_rate=0.00044682753699024145`, `learning_starts=1000`, `eps_start=0.059769600826486025`, `num_episodes=500`, and `ground_thrust_penalty=0.05346038291108023`, with the remaining DQN HPs inherited from the 253er baseline.

The useful step was not a full retrain. It was targeted refinement: start from the 253er checkpoint, use a fresh optimizer and replay buffer, lower `eps_start`, keep training short, and apply a mild ground-thrust penalty. The best penalty value was close to the lower edge of the search space (`~0.053`), suggesting Elise needed a small nudge, not a hard constraint.

==Mental model: Preserve a strong pilot, identify one concrete weakness, and fine-tune the checkpoint into the next model generation.==


## A12 How To Build A Small Good Five-World Lander

Ein kleiner guter 5-Welten-Lander braucht nicht zuerst ein riesiges Gehirn, sondern die richtigen Sinne, genug passende Flugstunden und fein abgestimmte Lernmechanik. Der 253er hat nur `10` Inputs, zwei Hidden Layers mit je `128` Neuronen, also `256` Hidden-Neuronen, und `4` Actions.

Mental model: kleines Gehirn, gute Hebel, gutes Popometer, viele Flugstunden, fein getrimmtes Fahrwerk.

Praktisch hieß das: 10D mit `dv_x/dt` und `dv_y/dt`, stark weltgewichtetes Training, `num_episodes=2000`, `batch_size=512`, `optimize_every=2`, `learning_starts=2500`, und: "ich lass jetzt Optuna mal sein Ding machen." Optuna durfte ziemlich viele HPs gleichzeitig anfassen und fand durch Fummeln an `gamma=0.995` und `tau=0.002` den empfindlichen HP-Korridor. Ergebnis: ein konserviertes Elise-Modell mit `252.6` Gym Score und allen fünf Welten über `200`.

Das Dashboard war der Wegweiser: die farbigen Training-Dots zeigten früh, dass Earth und Venus als blaue/gelbe Punkte mit niedrigem Score entlang der x-Achse hängen blieben. Ohne diese Sicht wäre die richtige Gegenmaßnahme - mehr eigene Flugstunden und später das Popometer - viel schwerer zu erkennen gewesen.

> **Essenz:** Nutze Hebel und entwickle Spezialwerkzeug. Dann ergänze ein Popometer.

* Die Hebel: Codex zum Denken und Modellieren, Optuna zum Suchen im empfindlichen und hochkomplexen HP-Raum.
* Das selbstentwickelte Spezialwerkzeug: Das Dashboard, um in der gewaltigen Datenmenge Wesentliches zu sehen.

> [!quote] Codex:
> Das Dashboard ist ein Natural-Intelligence-Fokussierer: Es macht die HPO-Geschichte sichtbar genug, damit menschliche Intuition daran arbeiten kann.

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
