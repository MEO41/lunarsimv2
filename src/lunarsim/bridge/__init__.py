# SPDX-License-Identifier: BSD-3-Clause
"""Software-in-the-loop bridge (milestone 7, scenario S10).

The ``ros2_bridge`` bindings (``nav: ros2_bridge``, ``agent: ros2_bridge``) keep the interface of
every other binding. They send the inner binding's inputs to a separate node process, which runs
the unchanged inner binding (for example ``ekf_fdi`` or ``ppo_gru`` with its ONNX export), and
take its output back. So the GNC code leaves the simulator with no change to the other slots.

- :mod:`lunarsim.bridge.message_transport`: how requests reach a node (``loopback``: a separate
  process over pipes, works everywhere; ``ros2``: rclpy topics, needs a ROS 2 installation);
- :mod:`lunarsim.bridge.node_programs`: what a node runs (a navigation binding or an agent);
- :mod:`lunarsim.bridge.onnx_policy`: ONNX export of a trained policy and a runner for it;
- :mod:`lunarsim.bridge.deadline_model`: message latency, executor jitter and the deadline.
"""
