# Lessons Learned

## Visualize Early

**Lesson:** Live diagrams are not decoration; they are diagnostic tools.

**Example:** The colored current-trial plot made visible that Earth and Venus are the hard worlds.

**Consequence:** Keep improving dashboard diagnostics and make training behavior visible before guessing what is wrong.

## Let Optuna Explore

**Lesson:** Optuna is not just plumbing; broad search can produce real breakthroughs.

**Example:** The 9D Earth-only breakthrough happened after letting Optuna optimize several HPs at once in a wider search space.

**Consequence:** Do not narrow the search too early. When the situation is unclear, deliberately let Optuna do its thing.

## Back Up Immediately

**Lesson:** A good checkpoint only counts once it is preserved.

**Example:** The observed `211` and `242` pilots showed what was possible, but their concrete checkpoints were not saved in time.

**Consequence:** Automatically copy new best eval checkpoints to Drive.
