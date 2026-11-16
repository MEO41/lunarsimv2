# SPDX-License-Identifier: BSD-3-Clause
"""S8 smoke test (GPU, milestone 5): Bekker footpad soil, L1 plume and the dust fault run together.

16 landers fly S8 with ZEM/ZEV. Checks: every record has the dust and soil columns; the footpads
sink into the soil (no PhysX footpad contact) and the landers still settle; the plume raises dust
below the erosion onset height; only truth-side bindings read TruthState.
"""

from isaac_test_support import build_agent, collect_episodes, make_environment


def test_soil_plume_and_dust_columns():
    """Landers settle on the soil, sink a few centimetres or more, and fly through dust."""
    environment = make_environment("s8_plume_regolith", ["num_envs=16"])
    frame = collect_episodes(environment, build_agent(environment), episodes_per_environment=1, maximum_steps=5000)
    for column in ("dust_blinded_time_s", "erosion_start_nozzle_height_m", "ejecta_mass_kg", "peak_dust_density",
                   "plume_torque_at_contact_n_m", "footpad_sinkage_max_m", "footpad_sinkage_final_mean_m"):
        assert column in frame
    assert frame["success"].mean() > 0.8
    assert (frame["footpad_sinkage_final_mean_m"] > 0.01).all()
    assert (frame["peak_dust_density"] > 0.5).all()
    assert (frame["erosion_start_nozzle_height_m"] <= 30.5).all()
    readers = set(environment.bus.truth_read_log)
    assert readers <= {"TerminalSensorSuite", "DustMetrics", "PhysxSitePlant"}
    assert environment.bindings["plant"].soil_contact
    environment.close()
