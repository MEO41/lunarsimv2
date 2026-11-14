# SPDX-License-Identifier: BSD-3-Clause
"""Frame changes. This is the only module that may change the frame of a TruthState.

Milestone 0 stub: only the SITE frame exists. The Moon-centred frames and the handoff arrive
with the orbit plant in milestone 6.
"""

from __future__ import annotations

from lunarsim.core.message_types import Frame, TruthState


def convert_truth_state(state: TruthState, target_frame: Frame) -> TruthState:
    """Return ``state`` expressed in ``target_frame``.

    Same frame: the state is returned unchanged. Any real conversion raises until milestone 6.
    """
    if not isinstance(target_frame, Frame):
        raise TypeError("target_frame must be a Frame")
    if state.frame is target_frame:
        return state
    raise NotImplementedError(
        f"Frame change {state.frame.value} -> {target_frame.value} is not implemented yet (milestone 6)"
    )
