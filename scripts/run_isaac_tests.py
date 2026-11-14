# SPDX-License-Identifier: BSD-3-Clause
"""Run the tests that need Isaac Lab and a GPU, one process per test file.

Usage::

    python scripts/run_isaac_tests.py                 # tests/physics and tests/smoke
    python scripts/run_isaac_tests.py physics         # one folder
    python scripts/run_isaac_tests.py tests/smoke/test_null_loop_environment.py

One simulator process can hold only one environment, so every file gets a fresh process.
"""

import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def collect_test_files(selections: list[str]) -> list[Path]:
    """Return the test files that match the command-line selections."""
    if not selections:
        selections = ["physics", "smoke"]
    files: list[Path] = []
    for selection in selections:
        path = Path(selection)
        if not path.exists():
            path = REPOSITORY_ROOT / "tests" / selection
        files += sorted(path.glob("test_*.py")) if path.is_dir() else [path]
    return files


def main() -> int:
    """Run each file with pytest in its own process and print a summary table."""
    results = []
    for test_file in collect_test_files(sys.argv[1:]):
        start_time = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        summary_lines = [line for line in completed.stdout.splitlines() if " passed" in line or " failed" in line or " error" in line]
        summary = summary_lines[-1].strip("= ") if summary_lines else f"no summary (exit code {completed.returncode})"
        results.append((test_file.relative_to(REPOSITORY_ROOT), completed.returncode, summary, time.perf_counter() - start_time))
        status = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"[{status}] {results[-1][0]}  {summary}  ({results[-1][3]:.0f} s)", flush=True)
        if completed.returncode != 0:
            failure_lines = [line for line in completed.stdout.splitlines() if line.startswith(("E ", "FAILED", "ERROR")) or "Error" in line]
            print("\n".join(failure_lines[-40:]), flush=True)
    failed = [result for result in results if result[1] != 0]
    print(f"\n{len(results) - len(failed)} of {len(results)} test files passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
