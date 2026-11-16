# SPDX-License-Identifier: BSD-3-Clause
"""Footprint hazard assessment (Apollo LM design surface, NASA TN D-6850 p. 3)."""

import math

import torch

from lunarsim.terrain.hazard_assessment import FootprintAssessor, HazardCriteria, candidate_patches, patch_size_for

CELL = 0.25


def assessor() -> FootprintAssessor:
    """Return an assessor with the LM criteria on a 0.25 m grid."""
    return FootprintAssessor(HazardCriteria(), CELL)


def test_flat_ground_is_safe_with_zero_slopes_and_relief():
    """A flat, fully known footprint passes every criterion."""
    judge = assessor()
    result = judge.assess(torch.zeros(1, judge.size, judge.size))
    assert bool(result["safe"][0])
    assert float(result["mean_slope_deg"][0]) < 1e-3
    assert float(result["effective_slope_deg"][0]) < 1e-3
    assert float(result["relief_m"][0]) < 1e-6
    assert abs(float(result["coverage"][0]) - 1.0) < 1e-6


def test_tilted_plane_gives_its_slope_and_fails_above_six_degrees():
    """A plane tilted 4 deg is safe, one tilted 8 deg is not (mean slope limit 6 deg)."""
    judge = assessor()
    for angle, expect_safe in ((4.0, True), (8.0, False)):
        heights = math.tan(math.radians(angle)) * judge.offset_x
        result = judge.assess(heights.unsqueeze(0))
        assert abs(float(result["mean_slope_deg"][0]) - angle) < 0.05
        assert abs(float(result["effective_slope_deg"][0]) - angle) < 0.3
        assert float(result["relief_m"][0]) < 1e-4  # a plane has no relief about itself
        assert bool(result["safe"][0]) is expect_safe


def test_tall_rock_in_the_middle_breaks_the_relief_limit_only():
    """A 0.8 m rock under the body (not under a pad) exceeds the 24 in relief, not the pad plane."""
    judge = assessor()
    heights = torch.zeros(1, judge.size, judge.size)
    centre = (judge.size - 1) // 2
    heights[0, centre - 2 : centre + 3, centre - 2 : centre + 3] = 0.8
    result = judge.assess(heights)
    assert float(result["relief_m"][0]) > 0.61
    assert float(result["effective_slope_deg"][0]) < 1.0
    assert not bool(result["safe"][0])


def test_rock_under_one_pad_tilts_the_pad_plane():
    """A 0.6 m rock under the +x pad gives an effective slope near atan(0.6 / (2 * 4.26 m)) or more."""
    judge = assessor()
    heights = torch.zeros(1, judge.size, judge.size)
    radius = judge.criteria.footpad_radius_from_axis_m
    under_pad = (torch.hypot(judge.offset_x - radius, judge.offset_y) < 0.6)
    heights[0][under_pad] = 0.6
    result = judge.assess(heights)
    assert float(result["effective_slope_deg"][0]) > math.degrees(math.atan(0.6 / (2.0 * radius))) - 0.5


def test_unknown_cells_lower_coverage_and_unsafe_when_too_little_is_seen():
    """With only a quarter of the disk known, the footprint is not safe even on flat ground."""
    judge = assessor()
    heights = torch.zeros(1, judge.size, judge.size)
    known = torch.zeros_like(heights, dtype=torch.bool)
    known[0, : judge.size // 2, : judge.size // 2] = True
    result = judge.assess(heights, known)
    assert float(result["coverage"][0]) < 0.3
    assert not bool(result["safe"][0])
    assert float(result["margin"][0]) == -1.0


def test_candidate_patches_cut_grids_into_centred_footprints():
    """Patches of a grid holding x (cell column) are centred on every stride-th cell."""
    patch = patch_size_for(HazardCriteria(), 0.5)
    width = patch + 8
    columns = torch.arange(width, dtype=torch.float32).repeat(width, 1).unsqueeze(0)  # value = column index
    known = torch.ones_like(columns, dtype=torch.bool)
    values, masks, rows, cols = candidate_patches(columns, known, patch, 4)
    assert (rows, cols) == (3, 3)
    centre = (patch - 1) // 2
    assert float(values[0, centre, centre]) == centre
    assert float(values[1, centre, centre]) == centre + 4
    assert bool(masks.all())
