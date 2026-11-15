# SPDX-License-Identifier: BSD-3-Clause
"""Copy the results of finished runs from ``runs/`` (not in git) into ``results/`` (in git).

User rule (2026-11-15): after every run, its results are committed, so that the commits show
which code change is responsible for which result. Example::

    python scripts/archive_run_results.py                 # every finished run not archived yet
    python scripts/archive_run_results.py runs/ppo_mlp_s3_seed0 runs/grid_zem_zev_ekf

A run is finished when it has ``manifest.yaml`` and either ``policy.pt`` (a training, written
after the last iteration) or ``episodes.parquet`` (an evaluation, written at the end). For each
run, ``results/<run name>/`` gets:

- ``manifest.yaml``: the full resolved scenario, seeds and the code commit the run was made with;
- ``summary.yaml``: run kind, code commit, scenario, agent, seed and the headline numbers;
- evaluations: ``episodes.parquet`` (one row per episode);
- trainings: ``policy.pt`` (the exported actor) and ``training_curve.csv`` (every logged
  scalar per iteration, from the TensorBoard file);
- ``log.txt.gz``: the console log of the run, when ``runs/`` holds one whose name matches.

Checkpoints (``model_*.pt``) and trajectory files stay in ``runs/``: they are large, and the
policy plus the commit reproduce them. ``results/index.csv`` lists every archived run.
A run already in ``results/`` is skipped unless ``overwrite=true``. Options: ``results=<directory>``
(where to write, default ``results``; ``scripts/push_run_results.sh`` points it into its own
clone), ``index_only=true`` (only rebuild the index).
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

import pandas
import yaml

RUNS = Path("runs")
INDEX_COLUMNS = ("run", "kind", "started_utc", "code_commit", "uncommitted_changes", "scenario", "agent", "navigation", "seed",
                 "episodes", "success_percent", "final_training_success_percent")


def run_kind(run: Path) -> str | None:
    """Return 'training', 'evaluation', or None if the run is not finished (or not a run)."""
    if not (run / "manifest.yaml").is_file():
        return None
    if (run / "policy.pt").is_file():
        return "training"
    if (run / "episodes.parquet").is_file():
        return "evaluation"
    return None


def matching_log(run: Path) -> Path | None:
    """Return the console log in ``runs/`` whose name (without ``train_`` and ``.log``) is the longest prefix of the run name."""
    best = None
    for log in RUNS.glob("*.log"):
        stem = log.stem.removeprefix("train_")
        if run.name.startswith(stem) and (best is None or len(stem) > len(best.stem.removeprefix("train_"))):
            best = log
    return best


def training_curve(run: Path) -> pandas.DataFrame | None:
    """Return every scalar of the run's TensorBoard files as one row per iteration, or None without them."""
    files = sorted(run.glob("events.out.tfevents.*"))
    if not files:
        return None
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    columns: dict[str, dict[int, float]] = {}
    for file in files:
        accumulator = EventAccumulator(str(file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            if tag.endswith("/time"):
                continue  # the same values against wall time
            for event in accumulator.Scalars(tag):
                columns.setdefault(tag.strip("/"), {})[event.step] = event.value
    frame = pandas.DataFrame(columns).sort_index()
    frame.index.name = "iteration"
    return frame


def summarise(run: Path, kind: str, manifest: dict, curve: pandas.DataFrame | None) -> dict:
    """Return the summary record of a run (also one row of the index)."""
    scenario = manifest.get("resolved_scenario", {})
    git = manifest.get("git", {})
    agent = scenario.get("agent", {})
    summary = {
        "run": run.name,
        "kind": kind,
        "started_utc": manifest.get("created_utc", ""),
        "code_commit": git.get("commit", ""),
        "uncommitted_changes": git.get("uncommitted_changes", ""),
        "scenario": manifest.get("scenario_name", scenario.get("name", "")),
        "agent": agent.get("binding", "") + (f" ({agent['ppo'].get('model', 'mlp')})" if isinstance(agent.get("ppo"), dict) else ""),
        "navigation": scenario.get("nav", {}).get("binding", ""),
        "seed": manifest.get("root_seed", ""),
        "command_line": " ".join(str(manifest.get("command_line", "")).split()),
        "resumed_from": (manifest.get("extra") or {}).get("resumed_from") or "",
        "episodes": "",
        "success_percent": "",
        "final_training_success_percent": "",
    }
    if kind == "evaluation":
        episodes = pandas.read_parquet(run / "episodes.parquet")
        summary["episodes"] = len(episodes)
        summary["success_percent"] = round(100.0 * float(episodes["success"].mean()), 2)
        summary["outcomes"] = {str(key): int(value) for key, value in episodes["outcome"].value_counts().items()}
        summary["touchdown_vertical_speed_mps_mean"] = round(float(episodes["touchdown_vertical_speed_mps"].mean()), 3)
        summary["propellant_used_kg_mean"] = round(float(episodes["propellant_used_kg"].mean()), 1)
    if curve is not None and "lunarsim/success_rate" in curve:
        last = curve["lunarsim/success_rate"].dropna()
        summary["final_training_success_percent"] = round(100.0 * float(last.tail(10).mean()), 2)
        summary["iterations"] = int(curve.index.max()) + 1
    return summary


def archive(run: Path, results: Path, overwrite: bool) -> dict | None:
    """Copy one finished run into ``results/``; return its summary, or None when skipped."""
    kind = run_kind(run)
    target = results / run.name
    if kind is None or (target.exists() and not overwrite):
        return None
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "manifest.yaml", target / "manifest.yaml")
    with open(run / "manifest.yaml") as file:
        manifest = yaml.safe_load(file)
    curve = None
    if kind == "training":
        shutil.copy2(run / "policy.pt", target / "policy.pt")
        curve = training_curve(run)
        if curve is not None:
            curve.to_csv(target / "training_curve.csv", float_format="%.6g")
    else:
        shutil.copy2(run / "episodes.parquet", target / "episodes.parquet")
    log = matching_log(run)
    if log is not None:
        with open(log, "rb") as source, gzip.open(target / "log.txt.gz", "wb") as destination:
            shutil.copyfileobj(source, destination)
    summary = summarise(run, kind, manifest, curve)
    with open(target / "summary.yaml", "w") as file:
        yaml.safe_dump(summary, file, sort_keys=False)
    print(f"[archive] {run.name}: {kind}, commit {str(summary['code_commit'])[:7]}, "
          f"success {summary['success_percent'] or summary['final_training_success_percent']}")
    return summary


def write_index(results: Path) -> Path:
    """Rebuild ``index.csv`` from every ``summary.yaml`` under the results directory."""
    rows = []
    for path in sorted(results.glob("*/summary.yaml")):
        with open(path) as file:
            summary = yaml.safe_load(file)
        rows.append({column: summary.get(column, "") for column in INDEX_COLUMNS})
    index = results / "index.csv"
    pandas.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(index, index=False)
    return index


def main() -> None:
    """Archive the named runs, or every finished run in ``runs/``, then rebuild the index."""
    arguments = [argument for argument in sys.argv[1:] if "=" not in argument]
    options = dict(argument.split("=", 1) for argument in sys.argv[1:] if "=" in argument)
    overwrite = options.get("overwrite", "false").lower() == "true"
    results = Path(options.get("results", "results"))
    if options.get("index_only", "false").lower() == "true":
        print(f"[archive] index {write_index(results)}")
        return
    runs = [Path(argument) for argument in arguments] or sorted(path for path in RUNS.iterdir() if path.is_dir())
    archived = [summary for run in runs if (summary := archive(run, results, overwrite)) is not None]
    if archived:
        print(f"[archive] {len(archived)} run(s) archived; index {write_index(results)}")


if __name__ == "__main__":
    main()
