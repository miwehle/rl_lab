# HPO Research Notes

This folder keeps research knowledge separate from code and package documentation.

Research is cyclic: observations update hypotheses, and updated hypotheses guide the next questions and tests.

The HPO package is the executable model. Research updates both hypotheses and code.

The practical truth test is whether a change robustly increases checkpoint and HP scores, and preserves the good checkpoints it discovers.

> Motto: ==Correct is what robustly improves the Gym score.==

Short research loop:

```text
question -> hypothesis -> prediction -> test -> update
```

Three levels:

| Level | Question |
|---|---|
| Observation | What do we see? |
| Model | Why could this be happening? |
| Test | What would show that the model is wrong? |

- `observations.md`: Run observations, short interpretations, next steps, and empirical measurements.
- `aha.md`: Distilled Aha moments and lessons learned.
- `hypotheses.md`: Testable, action-guiding working hypotheses.
- `questions.md`: Open research questions before they become hypotheses.
- `ideas.md`: Loose ideas, future directions, and not-yet-prioritized research thoughts.
- `assets/`: Screenshots and other evidence linked from the notes.
