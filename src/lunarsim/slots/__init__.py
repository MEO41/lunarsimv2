# SPDX-License-Identifier: BSD-3-Clause
"""Slot bindings: one folder per slot, one file per binding.

Importing this package imports every binding module, which registers the bindings. Binding
modules import Isaac Lab only inside the methods that need the simulator, so this package
imports on a machine without Isaac (the registry check runs in CPU tests).
"""

from lunarsim.slots.actuators import engine_rcs_pulse as _actuators_engine_rcs_pulse  # noqa: F401
from lunarsim.slots.actuators import none as _actuators_none  # noqa: F401
from lunarsim.slots.faults import none as _faults_none  # noqa: F401
from lunarsim.slots.metrics import none as _metrics_none  # noqa: F401
from lunarsim.slots.metrics import touchdown as _metrics_touchdown  # noqa: F401
from lunarsim.slots.navigation import ekf as _navigation_ekf  # noqa: F401
from lunarsim.slots.navigation import truth as _navigation_truth  # noqa: F401
from lunarsim.slots.plant import physx_site as _plant_physx_site  # noqa: F401
from lunarsim.slots.plant import rigid_box as _plant_rigid_box  # noqa: F401
from lunarsim.slots.sensors import none as _sensors_none  # noqa: F401
from lunarsim.slots.sensors import terminal_suite as _sensors_terminal_suite  # noqa: F401
from lunarsim.slots.start import fixed as _start_fixed  # noqa: F401
from lunarsim.slots.start import terminal_box as _start_terminal_box  # noqa: F401
from lunarsim.slots.surface import flat as _surface_flat  # noqa: F401
from lunarsim.slots.surface import measured_terrain as _surface_measured_terrain  # noqa: F401
from lunarsim.slots.surface import procedural as _surface_procedural  # noqa: F401
