# Interaction Protocol

Short user phrases with stable meaning in this workspace.

`#go` / `GO`: implement the agreed change now. Keep the implementation focused on the current agreement.

`#ncy` / `NCY`: no change yet. Discuss, inspect, or draft only; do not edit workspace files.

`#focus`: answer only the specific question, compactly. Avoid extra teaching, broad context, or adjacent ideas unless needed to avoid a misleading answer.

`#teach`: explain with helpful context, intuition, tradeoffs, and, when useful, a mental model. Prefer explanations that help the user build reusable intuition, while staying relevant to the question.

Notebook cell metadata: when the user asks to "überarbeite das Notebook nach dem Schema mit export und requires" or similar, add compact headers to relevant code cells:

```python
# cell: short-stable-label
# requires: dependency-label-or-exported-name
# export: names_for_later_cells
```

Use `# cell:`, `# requires:`, and `# export:` consistently with colons. Keep labels short, self-explanatory, and easy to find with text search.
