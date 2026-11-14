# Claude Code System Rules for this Repository

1. **NO CRYPTIC ABBREVIATIONS:** Never use pseudo-names, acronyms, or shortenings for variables, classes, or files (e.g., do NOT use `M3`, `S2`, `HR7`). Use fully descriptive, verbose names (e.g., `MotorController3`, `SensorModule2`, `HoverRoutineTrajectory7`).
2. **PLAN BEFORE CODING:** Before making any file modifications, outline the exact files you will touch and the changes you will make. Wait for the user to say "Proceed" unless instructed otherwise.
3. **EXPLAIN AFTER EXECUTION:** After completing a file modification, provide a 2-sentence summary of what was actually changed.
4. **COMMENTING:** Add descriptive docstrings to every new function explaining what it does so the human developer can read the code later.

## Project memory

At the start of every session, read `docs/knowledge/README.md` and the notes it lists first (rules, pending decisions, status, machine setup). Update the matching notes in the same commit as the work and run `lunarsim knowledge-graph`.
