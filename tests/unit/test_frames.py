# SPDX-License-Identifier: BSD-3-Clause
"""Frames stub: same-frame conversion is the identity; others wait for milestone 6."""

import pytest
from helpers import make_truth_state

from lunarsim.core.frames import convert_truth_state
from lunarsim.core.message_types import Frame


def test_same_frame_conversion_returns_the_same_state():
    """SITE -> SITE changes nothing."""
    state = make_truth_state()
    assert convert_truth_state(state, Frame.SITE) is state


def test_other_frames_are_not_implemented_yet():
    """A real frame change must fail loudly until the orbit plant exists."""
    with pytest.raises(NotImplementedError):
        convert_truth_state(make_truth_state(), Frame.MOON_CENTERED_INERTIAL)
    with pytest.raises(TypeError):
        convert_truth_state(make_truth_state(), "SITE")
