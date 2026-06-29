# Key HPO Insights

## 1. Earth Is Learnable

**These:** Earth is learnable with 9D observations and suitable HPs.

**Evidence:** The Earth-only `s7_exploration` found several trials above `200` Gym score early, and later an incumbent around `230`.

**Consequence:** Earth is not a physical no-go for the small DQN; the earlier five-world weakness likely comes from the training setup.

## 2. Hard Worlds Need Flight Hours

**These:** Hard worlds need many own training episodes.

**Evidence:** Strong Earth trials only reached high training level very late; `160+` mean over 100 episodes appeared around episode `900` or later in the best trusted runs.

**Consequence:** Five-world training with `1000` total episodes is too short for Earth and Venus if worlds are sampled uniformly.

## 3. Sampling Should Favor Hard Worlds

**These:** Multi-world training needs world-dependent sampling rates.

**Evidence:** With uniform sampling, each world gets only about `num_episodes / 5`; reaching Earth-only-like exposure would require about `5000` total episodes, which is too expensive.

**Consequence:** Earth and Venus should appear more often in the training world list instead of only increasing `num_episodes`.

## 4. Observation Mode Is Still Open

**These:** 9D is the strongest current path, but 8D, 9D, and 11D are not fairly settled yet.

**Evidence:** 9D works on Earth. Earlier 8D and 11D comparisons were likely distorted by weak HPs and too short training.

**Consequence:** Continue with 9D pragmatically, then compare 8D, 9D, and 11D again with stronger HPs.

## 5. Good HPs Are Not Enough

**These:** Good HPs do not guarantee a good concrete model.

**Evidence:** Model quality varies strongly by training seed, and training-checkpoint score can diverge from greedy evaluation score. In Earth-only `s7_exploration`, an optimize trial reached about `242` greedy eval score, but the concrete checkpoint was not preserved; the best currently saved Earth checkpoint from that run is around `206`.

**Consequence:** Save and evaluate concrete good checkpoints immediately; BI11 remains central, and automatic Drive preservation of new best eval checkpoints is a core requirement.
