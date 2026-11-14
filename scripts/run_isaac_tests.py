# SPDX-License-Identifier: BSD-3-Clause
"""Run the tests that need Isaac Lab and a GPU, one process per test file.

Usage::

    python scripts/run_isaac_tests.py                 # tests/physics and tests/smoke
    python scripts/run_isaac_tests.py physics         # one folder
    python scripts/run_isaac_tests.py tests/smoke/test_null_loop_environment.py

One simulator process can hold only one environment, so every file gets a fresh process.
"""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FILE_TIME_LIMIT_SECONDS = 600
STARTUP_SILENCE_SECONDS = 120
LAUNCH_ATTEMPTS = 3
SHARED_MEMORY_DIRECTORY = Path("/dev/shm")
# Name patterns of the shared-memory files that the simulator leaves behind, ending in its process id.
SIMULATOR_SHARED_MEMORY_PATTERNS = ("carb-RStringInternals-*", "sem.carb-RStringInternals-*")


def process_is_alive(process_id: int) -> bool:
    """Return True if a process with this id exists."""
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale_simulator_files() -> int:
    """Remove the simulator's shared-memory files whose process has ended; return how many.

    Every simulator process leaves such files; after many runs a new launch can stall at start.
    Only files of dead processes are removed.
    """
    removed = 0
    for pattern in SIMULATOR_SHARED_MEMORY_PATTERNS:
        for path in SHARED_MEMORY_DIRECTORY.glob(pattern):
            process_id = path.name.rsplit("-", 1)[-1]
            if process_id.isdigit() and not process_is_alive(int(process_id)):
                path.unlink(missing_ok=True)
                removed += 1
    return removed


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
    return [file.resolve() for file in files]


def run_one_file(test_file: Path) -> tuple[int, str, list[str]]:
    """Run pytest on one file in its own process. Return (exit code, summary line, output lines).

    The simulator sometimes does not exit after pytest has finished. So the result is taken
    from pytest's summary line, and a process that is still alive 30 s after that line (or
    after ``FILE_TIME_LIMIT_SECONDS`` in total) is stopped. The simulator also sometimes stalls
    at launch and prints nothing; such a launch is stopped after ``STARTUP_SILENCE_SECONDS``
    and tried again, up to ``LAUNCH_ATTEMPTS`` times.
    """
    for attempt in range(1, LAUNCH_ATTEMPTS + 1):
        remove_stale_simulator_files()
        result = _run_once(test_file)
        if result is not None:
            return result
        print(f"[RETRY] {test_file.name}: the simulator printed nothing for {STARTUP_SILENCE_SECONDS} s "
              f"(attempt {attempt} of {LAUNCH_ATTEMPTS})", flush=True)
    return 1, f"the simulator did not start in {LAUNCH_ATTEMPTS} attempts", []


def _run_once(test_file: Path) -> tuple[int, str, list[str]] | None:
    """Run the file once. Return None if the simulator stalled silently at launch."""
    process = subprocess.Popen(
        [sys.executable, "-m", "pytest", str(test_file), "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        # Unbuffered, so the launch watchdog sees the simulator's first lines when they are printed.
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines: list[str] = []
    summary = ""
    reader_done = threading.Event()
    first_line = threading.Event()

    def read_output() -> None:
        """Collect the output and remember pytest's summary line."""
        nonlocal summary
        for line in process.stdout:
            first_line.set()
            lines.append(line.rstrip())
            if re.search(r"\b\d+ (passed|failed|error)", line) and " in " in line:
                summary = line.strip("= \n")
                reader_done.set()
        reader_done.set()

    threading.Thread(target=read_output, daemon=True).start()
    if not first_line.wait(timeout=STARTUP_SILENCE_SECONDS) and process.poll() is None:
        process.kill()
        process.wait()
        return None
    reader_done.wait(timeout=FILE_TIME_LIMIT_SECONDS)
    try:
        process.wait(timeout=30 if summary else 1)
    except subprocess.TimeoutExpired:
        process.kill()
    if not summary:
        return 1, f"no pytest summary within {FILE_TIME_LIMIT_SECONDS} s", lines
    failed = "failed" in summary or "error" in summary
    return (1 if failed else 0), summary, lines


def main() -> int:
    """Run each file with pytest in its own process and print a summary table."""
    results = []
    for test_file in collect_test_files(sys.argv[1:]):
        start_time = time.perf_counter()
        exit_code, summary, lines = run_one_file(test_file)
        results.append((test_file.relative_to(REPOSITORY_ROOT), exit_code))
        status = "PASS" if exit_code == 0 else "FAIL"
        print(f"[{status}] {results[-1][0]}  {summary}  ({time.perf_counter() - start_time:.0f} s)", flush=True)
        if exit_code != 0:
            failure_lines = [line for line in lines if line.startswith(("E ", "FAILED", "ERROR")) or "Error" in line]
            print("\n".join(failure_lines[-40:]), flush=True)
    failed = [result for result in results if result[1] != 0]
    print(f"\n{len(results) - len(failed)} of {len(results)} test files passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
