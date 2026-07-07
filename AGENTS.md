# Agent Notes

## DQN Testing

Run DQN tests from the repository root with the DQN virtual environment:

```powershell
dqn\.venv\Scripts\python.exe -m pytest dqn\tests
```

## HPO Package Context

For HPO work, read `hpo/codex_memory/package_definition.md` first.
It is the compact package definition meant to restore context at the start of a
new session, with supporting PlantUML diagrams in the same folder.
For established shorthand in this collaboration, also read
`hpo/codex_memory/interaction_protocol.md`.
For HPO research notes, current hypotheses, and experiment lessons, start with
`hpo/research/README.md`.

## Working Style

### KISS and YAGNI

KISS and YAGNI are high values in this workspace.
Do not add code just because it might help later.
Before adding code, check: is it required now, is there a simpler existing
pattern, and will it stay easy to understand tomorrow?

Keep modules easy to read and low in mental load.
Ask before adding complexity whose value is unclear.
Favor simplicity for flexibility and design for change: understandability is a
long-lived value, complexity must earn its cost, and changeability often comes
from simplicity rather than anticipation.

For design sketches and diagrams, start with the smallest useful representation.
Show only the core idea first; add detail only when the user asks for it or when
the current task clearly requires it.

Use abstractions only when they reduce current complexity or match an existing
pattern. Treat speculative generalization, optional modes, config flags,
future-proofing, and indirection as complexity costs and mental load.
Add them only for a concrete current need or when the user explicitly agrees.

### Explicit Simplification Tasks

When the task is explicitly to simplify, set a LOC budget before implementation: production-code diff should be net negative unless the user explicitly accepts a tradeoff.
If a proposed or emerging change makes production code grow, stop and ask before continuing.
Do not bundle simplification with new semantics, new data flows, or extra reporting fields unless the user explicitly agrees.

### Markdown

Keep each prose paragraph on one source line; use line breaks only for structure.

### Test Code

For structure and naming of new test code, follow `../nmt_lab/translator/AGENTS.md`, section `## Struktur und Benennung von Testcode`.
Existing tests in `rl_lab` do not need to be retrofitted just to match that section.
New tests in `reward_shaping` should follow it from the start.

### PlantUML

Keep `.puml` sequence diagrams compact and navigable.
Use `hide footbox` in sequence diagrams.
Use plain lifeline heads plus `url of Alias is [[...]]` for clickable participants, not inline-linked participant labels.
Use message links only when they point to one concrete function or API, and keep at most one link per message.
Prefer local `vscode://file/...` links for workspace code; use external API links sparingly for established external concepts such as Optuna, Gymnasium, and PyTorch.
Centralize the local workspace URI in `puml_links.iuml` and use `WORKSPACE_ROOT/...` in PUML links instead of repeating the absolute path.
Do not add extra notes just to hold secondary links.
Use `\n` in long labels when it keeps the diagram narrow and the link still works.
Use `update pumls` or `aktualisiere pumls` as an explicit maintenance command.
For that command, find all `.puml` files and refresh local `vscode://file/...:line:col` links semantically against the current workspace code.
Do not check all PUML links on every code change; do it only when this command is requested or when directly editing a PUML file.
After refreshing links, mechanically check that local targets exist, line numbers are valid, and messages do not contain multiple links.

### Alignment

Before changing workspace files, confirm alignment with the user on the direction and size of the change.

NCY means "no change yet": do not edit workspace files. Understand it as a chat
shortcut for discussing and drafting the approach first; code changes may follow
only after the user agrees.

GO means "go ahead": the user wants the discussed change implemented. Proceed
with the agreed direction and scope, and keep the implementation focused.

## Essence 

KISS/YAGNI, so complexity does not grow out of "this might be useful later."

Alignment before edits, so the agent does not rush ahead.

Simplicity for maintainability, so the system is still understandable and easy
to change tomorrow.
