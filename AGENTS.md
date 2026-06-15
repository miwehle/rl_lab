# Agent Notes

## DQN Testing

Run DQN tests from the repository root with the DQN virtual environment:

```powershell
dqn\.venv\Scripts\python.exe -m pytest dqn\tests
```

## Working Style

Before changing workspace files, confirm that the user agrees with the direction
and size of the change. When unsure, ask first.

NCY means "no change yet": do not edit workspace files. Understand it as a chat
shortcut for discussing and drafting the approach first; code changes may follow
only after the user agrees.

GO means "go ahead": the user wants the discussed change implemented. Proceed
with the agreed direction and scope, and keep the implementation focused.

Keep modules easy to read and low in mental load.
Ask before adding complexity whose value is unclear.
Favor simplicity for flexibility and design for change: understandability is a
long-lived value, complexity must earn its cost, and changeability often comes
from simplicity rather than anticipation.

KISS and YAGNI are high values. Do not add code because it might help later.
Before adding code, check: is it required now, is there a simpler existing
pattern, and will it stay easy to understand tomorrow?

For design sketches and diagrams, start with the smallest useful representation.
Show only the core idea first; add detail only when the user asks for it or when
the current task clearly requires it.

Use abstractions only when they reduce current complexity or match an existing
pattern. Treat speculative generalization, optional modes/branches, config
flags, future-proofing, and indirection as complexity costs and mental load.
Add them only for a concrete current need or when the user explicitly agrees.

## Essence 

Agreement before edits, so the agent does not rush ahead.

KISS/YAGNI, so complexity does not grow out of "this might be useful later."

Simplicity for maintainability, so the system is still understandable and easy
to change tomorrow.

This is not meant to make the agent smaller or less capable. It is intended as
a quality guideline.
