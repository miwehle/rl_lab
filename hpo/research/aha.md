# HPO Insights

## Current Insights

## Earth Is Learnable

The 9D SolarSystemLander can learn Earth. The Earth-only `s7_exploration` produced several `200+` candidates and showed that the earlier Earth weakness was not a physical impossibility, but a training setup problem.

==Mental model: Earth and Venus do not need pity; they need flight hours.==

## Hard Worlds Need Their Own Flight Hours

Earth and Venus are hard because they need many own training episodes. In uniform five-world training, `1000` total episodes mean only about `200` per world, which is far below the Earth-only exposure that produced strong pilots.

## Good HPs Are Not Enough

Good HPs are stochastic producers, not concrete models. Training seed and checkpoint choice matter enough that a strong HP set can still produce weak pilots, so concrete good checkpoints must be saved and evaluated.

==Mental model: HPs are producers, not models.==

**Model quality depends strongly on the training seed.** Earlier Elise studies already showed this: one seed reached about `167` mean score over five worlds, while robust re-evaluation of the same HPs fell to about `113` or `92`.

## Observation Mode Is Not Settled

9D is the strongest current path because gravity helped on Earth without the questionable 11D wind/turbulence signals. But old 8D/11D comparisons used weaker HPs and shorter training, so 8D, 9D, and 11D still deserve a fair comparison later.

## Lessons Learned

## Visualize Early

The dashboard's colored training plot made the real problem visible: Earth and Venus were the hard worlds. Live plots are not decoration; they are diagnostic instruments.

==Mental model: The dashboard is our microscope.==

## Let Optuna Explore

The Earth breakthrough came after letting Optuna search several HPs at once in a wider space. When the situation is unclear, narrowing too early can hide the path.

## Back Up Immediately

The observed `211` and `242` pilots showed what was possible, but their concrete checkpoints were not saved in time. A good checkpoint only counts once it is preserved.

## Code Complexity Is Part Of The Experiment

This follows from the [HPO research motto](README.md): correct HPO work improves the Gym score and keeps the code simple.

A better HPO algorithm is not only one that improves the Gym score. It must also stay understandable enough to keep the research loop alive and fast; otherwise software aging slows the experiment down.

LOC is a cheap first proxy for code complexity. It is imperfect, especially in Python formatting, but it gives high bang for the buck as an early warning signal when small ideas become large code.
