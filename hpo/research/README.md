# HPO Research Notes

This folder keeps research knowledge separate from code and package documentation.

Research is cyclic: observations update hypotheses, and updated hypotheses guide the next questions and tests.

==The HPO package is the executable model.== Research updates both hypotheses and code.

The practical truth test is whether a change robustly increases checkpoint and HP scores, and preserves the good checkpoints it discovers.

> Motto: ==Correct is what==
> 1. ==robustly improves the Gym score and==
> 2. ==keeps the code simple.==

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
- `_details/`: Supporting detail files for observations, aha items, hypotheses, or questions. The leading `_` marks supporting material, not the main entry point.

## File Schema

Main research files use short stable item IDs plus an overview table at the top, for example `O7`, `A3`, `H5`, or `Q2`.

Use Obsidian-friendly links in overview tables:

```markdown
[[#A3 Gamma And Tau Shape Learning Dynamics\|A3]]
```

Keep the main item concise. Put long tables, raw DB analyses, screenshots, or detailed reasoning in `_details/` and link to them from the item when useful. Detail files do not need a strict 1:1 relationship with items; one detail file may support several observations.
