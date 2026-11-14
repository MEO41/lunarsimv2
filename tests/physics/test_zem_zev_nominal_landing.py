# SPDX-License-Identifier: BSD-3-Clause
"""Controller build-up step 4 (GPU): full ZEM/ZEV guidance lands from nominal starts.

"Nominal" is the upper part of the terminal box, where there is time to act:
footpad altitude 20-50 m, vertical speed -3 to 0 m/s, each horizontal velocity component
-2 to 2 m/s, tilt 0-10 degrees. 256 landers, one episode each, random starts from seed 0.

Pass criteria and why:
- Every episode is a success: inside the Apollo landing-gear envelope at contact, settled and
  upright afterwards. The full-range study shows that all failures start below 18 m with more
  than 3 m/s of descent rate, so the nominal region must be free of failures.
- Vertical speed at contact between 0.5 and 1.6 m/s (same reasoning as in the vertical
  descent test), horizontal speed below 0.6 m/s (half the 1.22 m/s limit), tilt below 3 degrees
  (half the 6 degree limit).
- The lander touches down within 3 m of the target: ZEM/ZEV steers to a gate above the
  target; the final descent only removes horizontal speed and does not steer position.
- Propellant below 250 kg of the 306 kg that are usable.
"""

import numpy
import pytest
from isaac_test_support import build_agent, collect_episodes, make_environment

COUNT = 256
NOMINAL_START = ["start.altitude_m=[20,50]", "start.v_vertical_mps=[-3,0]"]


@pytest.fixture(scope="module")
def episodes():
    """Fly one landing per environment from nominal starts."""
    environment = make_environment("s1_terminal_truth", [f"num_envs={COUNT}", "agent=zem_zev", *NOMINAL_START])
    agent = build_agent(environment)
    frame = collect_episodes(environment, agent, episodes_per_environment=1, maximum_steps=6000)
    yield frame
    environment.close()


def test_all_nominal_starts_land_successfully(episodes):
    """256 of 256 landings succeed."""
    print(episodes["outcome"].value_counts().to_string())
    assert len(episodes) == COUNT
    assert (episodes["outcome"] == "success").all()


def test_touchdown_has_margin_to_the_limits(episodes):
    """Contact speeds and tilt stay well inside the landing-gear envelope."""
    summary = episodes[["touchdown_vertical_speed_mps", "touchdown_horizontal_speed_mps", "touchdown_tilt_deg", "touchdown_tilt_rate_deg_per_s", "propellant_used_kg"]].describe().loc[["min", "mean", "max"]]
    print(summary.round(3).to_string())
    assert episodes["touchdown_vertical_speed_mps"].between(-1.6, -0.5).all()
    assert (episodes["touchdown_horizontal_speed_mps"] < 0.6).all()
    assert (episodes["touchdown_tilt_deg"] < 3.0).all()
    assert (episodes["propellant_used_kg"] < 250.0).all()


def test_landing_is_close_to_the_target(episodes):
    """Every touchdown point lies within 3 m of the target at the site origin."""
    miss = numpy.hypot(episodes["touchdown_position_x_m"], episodes["touchdown_position_y_m"])
    print("LANDING_MISS_M mean", round(float(miss.mean()), 3), "max", round(float(miss.max()), 3))
    assert (miss < 3.0).all()
