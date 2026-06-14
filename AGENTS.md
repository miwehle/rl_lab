# Agent Notes

Run DQN tests from the repository root with the DQN virtual environment:

```powershell
dqn\.venv\Scripts\python.exe -m pytest dqn\tests
```

Keep modules easy to read and low in mental load.
Ask before adding complexity whose value is unclear.
Prefer KISS and YAGNI. Do not generalize speculatively; add complexity only for
a concrete current need.

Before changing files in the workspace, make sure the user clearly agrees with
the change. When in doubt, ask first and agree on the direction and size of the
change, especially for larger or structural edits.
The user strongly prefers KISS and YAGNI: keep modules as simple as possible,
avoid speculative abstractions, optional extras, and bells and whistles, and do
not add complexity unless the current need is concrete and agreed.
