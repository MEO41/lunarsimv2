# SPDX-License-Identifier: BSD-3-Clause
"""Run manifest: everything needed to rebuild a result, stored as ``manifest.yaml``."""

from __future__ import annotations

import datetime
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

MANIFEST_FILE_NAME = "manifest.yaml"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _git(arguments: list[str], directory: Path) -> str:
    """Run a git command and return its trimmed output, or 'unknown' if git fails."""
    try:
        completed = subprocess.run(["git", *arguments], cwd=directory, capture_output=True, text=True, timeout=10)
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _package_version(name: str) -> str:
    """Return the installed version of a package, or 'not installed'."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _isaac_lab_commit() -> str:
    """Return the git commit of the Isaac Lab source checkout, if Isaac Lab is importable."""
    try:
        import isaaclab

        return _git(["rev-parse", "HEAD"], Path(isaaclab.__file__).resolve().parent)
    except ImportError:
        return "not installed"


def _gpu_name() -> str:
    """Return the name of the first CUDA device, or 'none'."""
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        return "none"


def collect_manifest(scenario: Any, slot_seeds: dict[str, int], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the manifest dictionary for a run of ``scenario`` (a ScenarioConfig)."""
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "scenario_name": scenario.name,
        "root_seed": scenario.root_seed,
        "slot_seeds": dict(slot_seeds),
        "git": {
            "commit": _git(["rev-parse", "HEAD"], REPOSITORY_ROOT),
            "uncommitted_changes": bool(_git(["status", "--porcelain", "--untracked-files=no"], REPOSITORY_ROOT)),
        },
        "versions": {
            "lunarsim": _package_version("lunarsim"),
            "isaac_sim": _package_version("isaacsim"),
            "isaac_lab": _package_version("isaaclab"),
            "isaac_lab_commit": _isaac_lab_commit(),
            "torch": _package_version("torch"),
            "python": sys.version.split()[0],
        },
        "gpu_name": _gpu_name(),
        "extra": dict(extra or {}),
        "resolved_scenario": scenario.resolved,
    }


def write_manifest(
    run_directory: Path | str, scenario: Any, slot_seeds: dict[str, int], extra: dict[str, Any] | None = None
) -> Path:
    """Write ``manifest.yaml`` into the run directory and return its path."""
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / MANIFEST_FILE_NAME
    with open(path, "w") as file:
        yaml.safe_dump(collect_manifest(scenario, slot_seeds, extra), file, sort_keys=False)
    return path


def read_manifest(run_directory: Path | str) -> dict[str, Any]:
    """Read ``manifest.yaml`` of a run."""
    with open(Path(run_directory) / MANIFEST_FILE_NAME) as file:
        return yaml.safe_load(file)
