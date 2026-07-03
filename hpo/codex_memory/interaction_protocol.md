# Interaction Protocol

Short user phrases with stable meaning in this workspace.

`#go` / `GO`: implement the agreed change now. Keep the implementation focused on the current agreement.

`#ncy` / `NCY`: no change yet. Discuss, inspect, or draft only; do not edit workspace files.

`#focus`: answer only the specific question, compactly. Avoid extra teaching, broad context, or adjacent ideas unless needed to avoid a misleading answer.

`#teach`: explain with helpful context, intuition, tradeoffs, and, when useful, a mental model. Prefer explanations that help the user build reusable intuition, while staying relevant to the question.

Notebook cell metadata: when the user asks to "ueberarbeite das Notebook nach dem Schema mit cell und requires" or similar, add compact headers only to relevant code cells:

```python
# cell: short-stable-label
# cell: dependent-label; requires: dependency-label-or-public-name
# cell: multi-dependent-label; requires: first-dependency, second-dependency
```

Use `# cell:` for cells worth finding, referring to, or selectively rerunning. Add `; requires:` on the same line only when the dependency is not obvious from the notebook order. Separate multiple dependencies with `, `. Do not use `# export:`; top-level names without a leading `_` are public by convention, while temporary names should use a leading `_`. Keep labels short, self-explanatory, and easy to find with text search.
