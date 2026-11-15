# SPDX-License-Identifier: BSD-3-Clause
"""S3 smoke test (GPU, milestone 3): faults, ekf_fdi and fault_response run together.

64 landers fly S3 with ZEM/ZEV. Checks: every episode record has the fault columns; every lander
drew a fault; faults that started mid-flight with a bias are detected; the navigation never
reads the truth or the fault status (truth readers are the truth-side bindings only).
"""

from isaac_test_support import build_agent, collect_episodes, make_environment

from lunarsim.slots.faults.fault_schedule import FAULT_CODE


def test_fault_columns_and_truth_readers():
    """Records carry the fault columns; only truth-side bindings read TruthState."""
    environment = make_environment("s3_faults", ["num_envs=64"])
    frame = collect_episodes(environment, build_agent(environment), episodes_per_environment=1, maximum_steps=6000)
    for column in ("fault_type", "fault_target", "fault_onset_time_s", "fault_detected", "fault_detection_delay_s", "false_alarm",
                   "navigation_position_error_rms_m", "navigation_maximum_height_error_m"):
        assert column in frame
    assert (frame["fault_type"] != FAULT_CODE["none"]).all()
    biased = frame[(frame["fault_type"] == FAULT_CODE["bias"]) & ~frame["fault_present_at_start"] & (frame["fault_onset_time_s"] >= 0)]
    assert biased["fault_detected"].all()
    readers = set(environment.bus.truth_read_log)
    assert readers <= {"TerminalSensorSuite", "FaultSchedule", "FaultResponseMetrics"}
    environment.close()
