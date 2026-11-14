# SPDX-License-Identifier: BSD-3-Clause
"""Compare the installed versions with the pinned ones and test that the GPU works.

Usage::

    python scripts/check_installation.py

Prints one line per check and exits with code 1 if a pinned version differs or the GPU is
not usable. It never starts the simulator, so it takes a few seconds.
"""

import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ISAAC_LAB_COMMIT = "e86463df8a8d8dff275e50fdcfd892e10ba32ec1"
EXPECTED_PYTHON = (3, 12)


def installed_version(package: str) -> str | None:
    """Return the installed version of a package, or None if it is not installed."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def read_pinned_versions() -> dict[str, str]:
    """Read ``name==version`` lines from requirements-pinned.txt (extras in brackets are dropped)."""
    pinned: dict[str, str] = {}
    for line in (REPOSITORY_ROOT / "requirements-pinned.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, version = line.split("==")
        pinned[name.split("[")[0].strip()] = version.strip()
    return pinned


def isaac_lab_commit() -> str | None:
    """Return the git commit of the installed Isaac Lab source tree, or None if it cannot be found."""
    try:
        import isaaclab

        source_directory = Path(isaaclab.__file__).resolve()
    except ImportError:
        return None
    for parent in source_directory.parents:
        if (parent / ".git").exists():
            result = subprocess.run(["git", "-C", str(parent), "rev-parse", "HEAD"], capture_output=True, text=True)
            return result.stdout.strip() or None
    return None


def check_gpu() -> list[tuple[bool, str]]:
    """Return the results of the GPU checks: CUDA available, a small matrix product runs."""
    import torch

    results = []
    available = torch.cuda.is_available()
    results.append((available, f"torch sees a GPU: {torch.cuda.get_device_name(0) if available else 'NO'}"))
    if available:
        capability = torch.cuda.get_device_capability(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        results.append((True, f"compute capability {capability[0]}.{capability[1]}, {memory_gb:.1f} GB memory"))
        try:
            product = torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")
            torch.cuda.synchronize()
            results.append((bool(torch.isfinite(product).all()), f"CUDA {torch.version.cuda}: matrix product runs"))
        except Exception as error:  # noqa: BLE001 - any CUDA failure is a failed check
            results.append((False, f"CUDA matrix product failed: {error}"))
    return results


def main() -> int:
    """Run all checks, print them, and return the process exit code."""
    checks: list[tuple[bool, str]] = []
    version_text = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(((sys.version_info.major, sys.version_info.minor) == EXPECTED_PYTHON, f"python {version_text} (pinned 3.12)"))
    for package, pinned in read_pinned_versions().items():
        found = installed_version(package)
        checks.append((found == pinned, f"{package} {found} (pinned {pinned})"))
    # The torch build string carries the CUDA version, for example 2.11.0+cu130.
    torch_found = installed_version("torch")
    checks.append((bool(torch_found and torch_found.startswith("2.11.0")), f"torch build {torch_found}"))
    commit = isaac_lab_commit()
    checks.append((commit == ISAAC_LAB_COMMIT, f"Isaac Lab commit {commit} (pinned {ISAAC_LAB_COMMIT[:7]})"))
    lunarsim = installed_version("lunarsim")
    checks.append((lunarsim is not None, f"lunarsim package {lunarsim}"))
    try:
        checks += check_gpu()
    except ImportError:
        checks.append((False, "torch is not installed"))
    for passed, text in checks:
        print(f"[{'ok' if passed else 'DIFFERENT'}] {text}")
    failed = [text for passed, text in checks if not passed]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks agree with the pinned setup.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
