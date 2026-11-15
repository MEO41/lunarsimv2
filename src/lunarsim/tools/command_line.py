# SPDX-License-Identifier: BSD-3-Clause
"""The ``lunarsim`` command. It needs no simulator.

``lunarsim figures <run_dir>``   make all figures of a run from its files
``lunarsim summary <run_dir>``   print the outcome table of a run
``lunarsim compare label=dir[,dir...] ...``  compare agents on the same held-out starts
``lunarsim knowledge-graph``     rebuild docs/knowledge/graph.json and graph_overview.md
``lunarsim knowledge-graph --check``  report broken links and out-of-date graph files
"""

from __future__ import annotations

import argparse


def main(arguments: list[str] | None = None) -> int:
    """Parse the command line and run the chosen sub-command."""
    parser = argparse.ArgumentParser(prog="lunarsim", description="LunarSim tools that work on run directories.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("figures", "make the figures of a run"), ("summary", "print the outcome table of a run")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("run_directory", help="directory with manifest.yaml and episodes.parquet")
    compare = commands.add_parser("compare", help="compare agents on the same held-out starts")
    compare.add_argument("groups", nargs="+", help="label=run_dir[,run_dir...] (for example ppo=runs/a,runs/b)")
    knowledge = commands.add_parser("knowledge-graph", help="rebuild or check the project knowledge graph")
    knowledge.add_argument("--check", action="store_true", help="only check; exit 1 on any problem")
    options = parser.parse_args(arguments)

    if options.command == "knowledge-graph":
        from lunarsim.tools.knowledge_graph import check_knowledge_graph, write_knowledge_graph

        if not options.check:
            for path in write_knowledge_graph():
                print(path)
        problems = check_knowledge_graph()
        for problem in problems:
            print(f"problem: {problem}")
        print(f"knowledge graph: {len(problems)} problem(s)")
        return 1 if problems else 0
    if options.command == "compare":
        from lunarsim.tools.agent_comparison import compare_groups, outcome_counts

        groups = {}
        for group in options.groups:
            label, _, directories = group.partition("=")
            groups[label] = [directory for directory in directories.split(",") if directory]
        print(compare_groups(groups).round(2).to_string(index=False))
        print()
        print(outcome_counts(groups).to_string())
        return 0
    if options.command == "figures":
        from lunarsim.tools.figures import make_figures

        for path in make_figures(options.run_directory):
            print(path)
    else:
        from lunarsim.tools.episode_records import read_episode_records

        episodes = read_episode_records(options.run_directory)
        print(f"episodes: {len(episodes)}   success rate: {episodes['success'].mean():.4f}")
        print(episodes["outcome"].value_counts().to_string())
        if "start_mass_kg" in episodes:
            from lunarsim.tools.failure_analysis import failure_table
            from lunarsim.tools.run_manifest import read_manifest

            print()
            print(failure_table(episodes, read_manifest(options.run_directory)).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
