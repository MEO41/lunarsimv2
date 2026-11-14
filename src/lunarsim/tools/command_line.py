# SPDX-License-Identifier: BSD-3-Clause
"""The ``lunarsim`` command. It needs no simulator.

``lunarsim figures <run_dir>``   make all figures of a run from its files
``lunarsim summary <run_dir>``   print the outcome table of a run
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
    options = parser.parse_args(arguments)

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
