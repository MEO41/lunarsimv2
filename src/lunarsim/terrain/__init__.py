# SPDX-License-Identifier: BSD-3-Clause
"""Real lunar terrain: height-map reading, small-scale surface detail, and a batched height lookup.

Nothing in this package needs Isaac Lab; the surface binding ``measured_terrain`` uses it to
build the ground mesh, and the plant and metrics use :class:`TerrainHeightField` on the GPU.
"""
