---
id: knowledge-graph-tooling
title: Knowledge graph tooling
type: component
status: current
updated: 2026-11-14
summary: How this knowledge graph is stored, checked and regenerated, and the rule to update it in the same commit as the work.
---

# Knowledge graph tooling

- Notes: `docs/knowledge/<note-id>.md`, one topic each, with a YAML header (`id`, `title`,
  `type`, `status`, `updated`, `summary`). The entry point is `docs/knowledge/README.md`.
- Links: `[[note-id]]`. A list line `- relation: [[note-id]]` names the relation; other links
  count as "mentions". Links inside code (fenced blocks or `inline code`) are ignored. The format also opens as an
  Obsidian vault.
- Types: hub, rule, design, milestone, component, decision, result, issue, plan, research,
  environment, history. Statuses: current, planned, waiting-for-user, open, superseded.
- Code: `src/lunarsim/tools/knowledge_graph.py`; command `lunarsim knowledge-graph` writes
  `graph.json` (all nodes and edges) and `graph_overview.md` (Mermaid diagram of the named
  relations plus a table of all notes). `--check` reports bad headers, broken links, notes that
  nothing links to, and generated files that are out of date.
- CI: `tests/unit/test_knowledge_graph.py` runs the same check, so a broken link fails the GitHub
  CPU-test run.

## Rule

Update the matching notes **in the same commit** as the work: [[decisions-log]] for decisions,
[[pending-user-decisions]] for questions, [[milestone-roadmap-and-status]] for progress, the
result notes for numbers, [[session-history]] for the timeline. Then run
`lunarsim knowledge-graph`. `PROGRESS.md` stays the short public status page.

- required by: [[rules-from-the-user]]
