# SPDX-License-Identifier: BSD-3-Clause
"""Episode records: one row per episode with every metric, stored as ``episodes.parquet``."""

from __future__ import annotations

from pathlib import Path

import pandas
from torch import Tensor

EPISODES_FILE_NAME = "episodes.parquet"
TRAJECTORIES_FILE_NAME = "trajectories.parquet"


def records_to_frame(records: dict[str, Tensor], outcome_names: tuple[str, ...] | None = None) -> pandas.DataFrame:
    """Turn a dictionary of equally long tensors into a table. Adds an ``outcome`` name column."""
    frame = pandas.DataFrame({name: values.numpy() for name, values in records.items()})
    if outcome_names is not None and "outcome_code" in frame:
        frame["outcome"] = [outcome_names[code] for code in frame["outcome_code"]]
    return frame


class EpisodeRecordWriter:
    """Collects record tables during a run and writes them once at the end."""

    def __init__(self, run_directory: Path | str, outcome_names: tuple[str, ...] | None = None) -> None:
        """Remember the run directory; nothing is written yet."""
        self.run_directory = Path(run_directory)
        self.outcome_names = outcome_names
        self._frames: list[pandas.DataFrame] = []

    def append(self, records: dict[str, Tensor]) -> None:
        """Add the records drained from the metrics binding (may be empty)."""
        if records:
            self._frames.append(records_to_frame(records, self.outcome_names))

    @property
    def number_of_episodes(self) -> int:
        """Return how many episode rows have been collected."""
        return sum(len(frame) for frame in self._frames)

    def to_frame(self) -> pandas.DataFrame:
        """Return all collected rows as one table."""
        return pandas.concat(self._frames, ignore_index=True) if self._frames else pandas.DataFrame()

    def write(self) -> Path:
        """Write ``episodes.parquet`` into the run directory and return its path."""
        self.run_directory.mkdir(parents=True, exist_ok=True)
        path = self.run_directory / EPISODES_FILE_NAME
        self.to_frame().to_parquet(path, index=False)
        return path


def read_episode_records(run_directory: Path | str) -> pandas.DataFrame:
    """Read ``episodes.parquet`` of a run."""
    return pandas.read_parquet(Path(run_directory) / EPISODES_FILE_NAME)
