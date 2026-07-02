# HPO Hypotheses

Each hypothesis should stay attackable: what should happen if it is true, and what would make us update or drop it?

| Nr | Hypothesis | Topics |
|---|---|---|
| [[#H1 Earth Is Learnable\|H1]] | Earth Is Learnable | SSL |
| [[#H2 Hard Worlds Need Flight Hours\|H2]] | Hard Worlds Need Flight Hours | SSL |
| [[#H3 Sampling Should Favor Hard Worlds\|H3]] | Sampling Should Favor Hard Worlds | SSL, Sampling |
| [[#H4 Observation Mode Is Still Open\|H4]] | Observation Mode Is Still Open | SSL |
| [[#H5 Good HPs Are Not Enough\|H5]] | Good HPs Are Not Enough | RL, Checkpointing |

Topics: `RL` = Reinforcement Learning, `SSL` = SolarSystemLander.

## H1 Earth Is Learnable

**These:** Earth is learnable with 9D observations and suitable HPs.

**Evidence:** The Earth-only `s7_exploration` found several trials above `200` Gym score, with the best observed optimize trial around `242` and the best preserved checkpoint around `206`. The useful HP region currently points to `num_episodes=1000`, `batch_size=512`, `eps_end~0.02..0.04`, and `eps_decay~31k..43k`. Evidence is still small, but the robustness plot suggests `learning_rate~7e-4..1e-3` may be more reliable than much higher values; the `4.5e-3` candidate reached a good optimize score but failed badly in robustness.

**Prediction:** Repeating Earth-only 9D studies near this HP region should keep producing `200+` pilots.

**Could be wrong if:** Further Earth-only runs near this region fail to reproduce `200+`, or the observed wins turn out to be mostly lucky seed outliers.

**Consequence:** Earth is not a physical no-go for the small DQN; the earlier five-world weakness likely comes from the training setup.

**When:** 2026-07-01.

## H2 Hard Worlds Need Flight Hours

**These:** Hard worlds need many own training episodes.

**Evidence:** Strong Earth trials only reached high training level very late; `160+` mean over 100 episodes appeared around episode `900` or later in the best trusted runs.

**Prediction:** Giving Earth and Venus more own episodes should improve their scores more than merely tuning small HP details.

**Could be wrong if:** Longer training does not improve Earth/Venus, or failures come mainly from model capacity, reward dynamics, or weather rather than exposure.

**Consequence:** Five-world training with `1000` total episodes is too short for Earth and Venus if worlds are sampled uniformly.

**When:** 2026-07-01.

## H3 Sampling Should Favor Hard Worlds

**These:** Multi-world training needs world-dependent sampling rates.

**Evidence:** With uniform sampling, each world gets only about `num_episodes / 5`; reaching Earth-only-like exposure would require about `5000` total episodes, which is too expensive.

**Prediction:** Oversampling Earth and Venus should improve five-world training without requiring `5000` total episodes.

**Could be wrong if:** Oversampling hurts the easy worlds too much, causes forgetting, or still fails to lift Earth/Venus.

**Consequence:** Earth and Venus should appear more often in the training world list instead of only increasing `num_episodes`.

**When:** 2026-07-01.

## H4 Observation Mode Is Still Open

**These:** 9D is the strongest current path, but 8D, 9D, and 11D are not fairly settled yet.

**Evidence:** 9D works on Earth. Earlier 8D and 11D comparisons were likely distorted by weak HPs and too short training.

**Prediction:** A fair comparison with stronger HPs will show whether gravity-only 9D is enough or whether 8D/11D can match or beat it.

**Could be wrong if:** 8D performs equally well with good HPs, or 11D improves once HPs and training length are corrected.

**Consequence:** Continue with 9D pragmatically, then compare 8D, 9D, and 11D again with stronger HPs.

**When:** 2026-07-01.

## H5 Good HPs Are Not Enough

**These:** Good HPs do not guarantee a good concrete model.

**Evidence:** Model quality varies strongly by training seed, and training-checkpoint score can diverge from greedy evaluation score. In Earth-only `s7_exploration`, an optimize trial reached about `242` greedy eval score, but the concrete checkpoint was not preserved; the best currently saved Earth checkpoint from that run is around `206`.

**Prediction:** Re-running the same HPs will keep producing a broad score distribution, so preserving concrete good checkpoints will matter more than trusting HPs alone.

**Could be wrong if:** Stronger evaluation and training settings make repeated runs with the same HPs consistently similar.

**Consequence:** Save and evaluate concrete good checkpoints immediately; BI11 remains central, and automatic Drive preservation of new best eval checkpoints is a core requirement.

**When:** 2026-07-01.
