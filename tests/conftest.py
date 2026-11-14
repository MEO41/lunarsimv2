# SPDX-License-Identifier: BSD-3-Clause
"""Make ``tests/`` importable (for isaac_test_support) from every test file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
