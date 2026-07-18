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

Before answering concrete factual questions, verify local facts first when a
few seconds of inspection can avoid hedged answers such as "if" or "probably."

### Design First

Find the good design first; only then discuss implementation details. Good design is the first practical lever for KISS.

KISS is king, and simple design is his closest ally.

If the design is questionable or not KISS, say so plainly. Always prioritize KISS. Do not leave this path, and help the user stay on it too.

If the user appears to be following an unnecessarily complex path, pause and point to the simpler design before implementing.

### Deep Modules and Detail Hiding

Prefer deep modules: separate the levers from the details. Prefer deep modules over shallow ones: hide real complexity behind small, clear interfaces. A good API should reduce the caller's cognitive load.

Deep modules and information hiding are very welcome in this workspace. Public APIs should make common use simple, expose the right levers, and keep implementation details under the hood.

### Refactoring and Automated Tests

KISS is king. Refactoring keeps the kingdom tidy; automated tests guard the gates. Treat refactoring as an active parallel process that preserves or restores simple design while behavior evolves, and use automated tests to keep that process safe.

Outside the user's explicit `#focus` mode, proactively point out concrete, likely worthwhile simplification or refactoring opportunities; ask whether to schedule or do them soon instead of silently carrying complexity forward.

### Cost-Aware Alternative Selection

When proposing or comparing implementation alternatives, estimate the likely cost and complexity of each option, especially hidden infrastructure cost such as orchestration, persistence, evaluation, reporting, notebooks, tests, and integration glue.

If the user chooses an option that appears significantly more complex or costly than another viable option, pause before implementation and explicitly call out the tradeoff. Ask for confirmation in plain language, for example: "This option likely costs much more code because it duplicates existing experiment infrastructure. Do you still want this path?"

Do not treat "go" as overriding this warning when the selected option conflicts with KISS/YAGNI or appears to create avoidable infrastructure duplication. First confirm that the user intentionally accepts the extra complexity.

### Valuable Artifacts

HPO best-eval checkpoints, especially Drive archives, are expensive and hard-won artifacts. Treat them carefully. Never delete, overwrite, rename, or replace them unless the user explicitly asks for that exact destructive action after being warned.

If a high-value checkpoint has an old or incompatible format, first create and verify a copied/migrated artifact beside it. After the new-format copy works, old-format artifacts do not need to be kept forever and may be deleted when the user intentionally chooses that. Runtime compatibility shims should still be avoided unless explicitly agreed.

### Explicit Simplification Tasks

When the task is explicitly to simplify, set a LOC budget before implementation: production-code diff should be net negative unless the user explicitly accepts a tradeoff.
If a proposed or emerging change makes production code grow, stop and ask before continuing.
Do not bundle simplification with new semantics, new data flows, or extra reporting fields unless the user explicitly agrees.

### Markdown

Keep each prose paragraph on one source line; use line breaks only for structure.

### Encoding

Use UTF-8 for text files. When reading German prose or other non-ASCII text through PowerShell, make the command umlaut-safe by setting console output encoding to UTF-8 and reading files as UTF-8, for example: `[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); $OutputEncoding = [System.Text.UTF8Encoding]::new(); Get-Content -Raw -Encoding UTF8 path\to\file.md`.

If umlauts or other non-ASCII text appear corrupted in tool output, assume an encoding/display issue first and verify with an UTF-8-safe read before treating it as a content problem.

### Test Code

For test code, follow `../nmt_lab/translator/how_to_test.md`. Treat it as part of this AGENTS.md.

Existing tests in `rl_lab` do not need to be retrofitted just to match those rules.

For HPO, distinguish two public API levels when applying the test rules:

- `api-public`: objects intended for notebooks and external clients. They should be re-exported from `hpo/__init__.py`.
- `module-public`: names without a leading `_` in a public module or public class. They may be used by higher-level package code without being re-exported from package `__init__.py` files.

In HPO, direct tests should usually target only one of these two public API levels. Names with a leading `_`, and members of a private surrounding structure, are private implementation details and should usually be tested through their public users instead.

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
