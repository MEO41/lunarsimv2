# SPDX-License-Identifier: BSD-3-Clause
"""Landing reward terms (CPU): outcome, touchdown quality, velocity tracking, attitude, propellant."""

import pytest
import torch
from helpers import make_truth_state

from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.core.message_types import EpisodeEvents
from lunarsim.env.reward_terms import DEFAULT_REWARD_WEIGHTS, LandingRewardTerms
from lunarsim.env.scenario_loader import default_config_directory

PRESET = load_lander_preset(default_config_directory() / "landers" / "apollo_lunar_module.yaml")
STEP = 0.02


def make_terms(count=1, **weights):
    """Return reward terms for ``count`` environments, all reset."""
    terms = LandingRewardTerms(weights, PRESET, count, "cpu", STEP)
    terms.reset(torch.arange(count))
    return terms


def truth(height=10.0, velocity=(0.0, 0.0, -2.5), mass=7000.0, position_xy=(0.0, 0.0)):
    """Return a one-environment upright truth state."""
    state = make_truth_state(1)
    state.position[:] = torch.tensor([*position_xy, height + 3.62])
    state.footpad_height_above_ground[:] = height
    state.velocity[:] = torch.tensor(velocity)
    state.mass[:] = mass
    return state


def events(finished=False, outcome=-1, first_contact=False, values=None):
    """Return one-environment episode events."""
    flag = torch.tensor([finished])
    return EpisodeEvents(
        terminated=flag,
        truncated=torch.tensor([False]),
        outcome_code=torch.tensor([outcome]),
        first_contact=torch.tensor([first_contact]),
        touchdown_values={name: torch.tensor([value]) for name, value in (values or {}).items()}
        or {name: torch.zeros(1) for name in ("vertical_speed", "horizontal_speed", "tilt", "tilt_rate")},
    )


def test_following_the_wanted_velocity_costs_nothing():
    """At 10 m the wanted descent is 10 / 4 = 2.5 m/s straight down: no tracking or attitude cost."""
    terms = make_terms()
    _, parts = terms.compute(truth(), events())
    assert float(parts["tracking"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(parts["attitude"][0]) == pytest.approx(0.0, abs=1e-6)


def test_hovering_costs_the_velocity_error_per_second():
    """Hovering at 10 m misses the wanted 2.5 m/s: cost 2.5 per second."""
    terms = make_terms()
    _, parts = terms.compute(truth(velocity=(0.0, 0.0, 0.0)), events())
    assert float(parts["tracking"][0]) == pytest.approx(-2.5 * STEP, rel=1e-5)


def test_outcomes_and_capped_touchdown_term():
    """Success +100; a failed touchdown -100; no touchdown -300; a crash costs at most 4 x 4 x 10."""
    terms = make_terms()
    reward, parts = terms.compute(truth(), events(finished=True, outcome=0))
    assert float(parts["outcome"][0]) == DEFAULT_REWARD_WEIGHTS["success"]
    terms = make_terms()
    _, parts = terms.compute(truth(), events(finished=True, outcome=5))
    assert float(parts["outcome"][0]) == DEFAULT_REWARD_WEIGHTS["failure_without_touchdown"]
    terms = make_terms()
    crash = {"vertical_speed": -30.0, "horizontal_speed": 10.0, "tilt": 1.0, "tilt_rate": 3.0}
    _, parts = terms.compute(truth(), events(finished=True, outcome=4, first_contact=True, values=crash))
    assert float(parts["outcome"][0]) == DEFAULT_REWARD_WEIGHTS["failure"]
    assert float(parts["touchdown"][0]) == pytest.approx(-4 * 4.0 * 10.0)


def test_touchdown_at_the_limits_costs_one_weight_per_value():
    """Touching down exactly at all four limits costs 4 x touchdown_weight."""
    terms = make_terms()
    limits = {
        "vertical_speed": -PRESET.maximum_vertical_touchdown_speed,
        "horizontal_speed": PRESET.maximum_horizontal_touchdown_speed,
        "tilt": PRESET.maximum_touchdown_tilt,
        "tilt_rate": PRESET.maximum_touchdown_tilt_rate,
    }
    _, parts = terms.compute(truth(), events(first_contact=True, values=limits))
    assert float(parts["touchdown"][0]) == pytest.approx(-40.0, rel=1e-5)


def test_no_tracking_after_contact_and_propellant_is_charged():
    """After the first contact only propellant counts: 2 kg burned costs 0.1."""
    terms = make_terms()
    terms.compute(truth(mass=7000.0), events(first_contact=True))
    _, parts = terms.compute(truth(velocity=(3.0, 0.0, 0.0), mass=6998.0), events())
    assert float(parts["tracking"][0]) == 0.0
    assert float(parts["propellant"][0]) == pytest.approx(-0.1)


def test_unknown_weight_is_rejected():
    """A misspelt reward weight stops the training before it starts."""
    with pytest.raises(ValueError):
        make_terms(sucess=1.0)
