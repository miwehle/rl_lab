# Agent Notes

Run DQN tests from the repository root with the DQN virtual environment:

```powershell
dqn\.venv\Scripts\python.exe -m pytest dqn\tests
```

Before changing workspace files, confirm that the user agrees with the direction
and size of the change. When unsure, ask first.

Keep modules easy to read and low in mental load.
Ask before adding complexity whose value is unclear.
Favor simplicity for flexibility and design for change: understandability is a
long-lived value, complexity must earn its cost, and changeability often comes
from simplicity rather than anticipation.

KISS and YAGNI are high values. Do not add code because it might help later.
Before adding code, check: is it required now, is there a simpler existing
pattern, and will it stay easy to understand tomorrow?

Use abstractions only when they reduce current complexity or match an existing
pattern. Treat speculative generalization, optional modes/branches, config
flags, future-proofing, and indirection as complexity costs and mental load.
Add them only for a concrete current need or when the user explicitly agrees.
