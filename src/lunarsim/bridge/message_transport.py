# SPDX-License-Identifier: BSD-3-Clause
"""How a bridge binding reaches its node: ``loopback`` (separate process) or ``ros2`` (rclpy topics).

Both transports carry the same payload: a request or reply pickled to bytes (CPU tensors inside).

``loopback`` (works everywhere, used by the tests): the node is a separate Python interpreter,
``python -m lunarsim.bridge.serve_node``, that connects back over a local socket (authenticated
with a random key). It never imports Isaac Sim. This gives real inter-process message timing
without a ROS 2 installation.

``ros2`` (written, **not tested**: ROS 2 is not installed on the machines of this project): the
request goes out on ``/lunarsim/<node name>/request`` and the reply comes back on
``/lunarsim/<node name>/reply``, both ``std_msgs/msg/UInt8MultiArray`` with the pickled bytes;
the simulator publishes its time on ``/clock`` (``rosgraph_msgs/msg/Clock``) so the nodes use sim
time (design S10: ``clock: sim_time``). The node side is ``python -m lunarsim.bridge.serve_node
--transport ros2``.
"""

from __future__ import annotations

import os
import pickle
import secrets
import subprocess
import sys
import time
from multiprocessing.connection import Listener
from typing import Any

START_TIMEOUT_S = 180.0


class BridgeNodeError(RuntimeError):
    """The node failed; the message holds the node's traceback."""


def encode(message: Any) -> bytes:
    """Return the message pickled to bytes (CPU tensors are stored by value)."""
    return pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)


def decode(payload: bytes) -> Any:
    """Return the message from its bytes."""
    return pickle.loads(payload)


class LoopbackTransport:
    """A node in a separate Python process, reached over a local socket."""

    def __init__(self, spec: dict, name: str) -> None:
        """Start the node process, send it the spec and wait until it reports ready."""
        self.name = name
        key = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=key)
        host, port = self.listener.address
        environment = dict(os.environ, LUNARSIM_BRIDGE_KEY=key.hex())
        self.process = subprocess.Popen(
            [sys.executable, "-m", "lunarsim.bridge.serve_node", "--transport", "loopback", "--address", f"{host}:{port}", "--name", name],
            env=environment,
        )
        self.connection = self._accept_within(START_TIMEOUT_S)
        self.connection.send_bytes(encode(spec))
        status, detail = decode(self.connection.recv_bytes())
        if status != "ready":
            self.close()
            raise BridgeNodeError(f"node {name} did not start:\n{detail}")

    def _accept_within(self, timeout_s: float):
        """Accept the node's connection, or fail if the node process exits or takes too long."""
        import socket

        self.listener._listener._socket.settimeout(timeout_s)
        try:
            return self.listener.accept()
        except socket.timeout as error:
            self.process.kill()
            raise BridgeNodeError(f"node {self.name} did not connect within {timeout_s} s (exit code {self.process.poll()})") from error

    def request(self, message: tuple) -> tuple[Any, float]:
        """Send one request and wait for its reply; return (reply, round trip in seconds)."""
        began = time.perf_counter()
        self.connection.send_bytes(encode(message))
        status, reply = decode(self.connection.recv_bytes())
        round_trip = time.perf_counter() - began
        if status != "ok":
            raise BridgeNodeError(f"node {self.name} failed on '{message[0]}':\n{reply}")
        return reply, round_trip

    def publish_clock(self, seconds: float) -> None:
        """Do nothing: a loopback node gets the time with every request."""

    def close(self) -> None:
        """Stop the node process."""
        try:
            self.connection.send_bytes(encode(("close",)))
        except Exception:
            pass
        try:
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()
        self.listener.close()


class Ros2Transport:
    """A node reached over ROS 2 topics (untested here; see the module text)."""

    def __init__(self, spec: dict, name: str, reply_timeout_s: float = 30.0) -> None:
        """Create the rclpy node, the topics and the clock publisher; send the spec as the first request."""
        try:
            import rclpy
            from rclpy.node import Node
            from rosgraph_msgs.msg import Clock
            from std_msgs.msg import UInt8MultiArray
        except ImportError as error:
            raise RuntimeError(
                "transport 'ros2' needs ROS 2 (rclpy, std_msgs, rosgraph_msgs) in this Python environment; "
                "use transport: loopback, or install ROS 2 and source its setup file"
            ) from error
        if not rclpy.ok():
            rclpy.init()
        self.rclpy, self.message_type, self.clock_type = rclpy, UInt8MultiArray, Clock
        self.name, self.reply_timeout_s = name, reply_timeout_s
        self.node = Node(f"lunarsim_bridge_{name}")
        self.publisher = self.node.create_publisher(UInt8MultiArray, f"/lunarsim/{name}/request", 10)
        self.clock_publisher = self.node.create_publisher(Clock, "/clock", 10)
        self.replies: dict[int, tuple] = {}
        self.node.create_subscription(UInt8MultiArray, f"/lunarsim/{name}/reply", self._on_reply, 10)
        self.sequence = 0
        status, detail = self._call(("spec", spec))
        if status != "ready":
            raise BridgeNodeError(f"node {name} did not start:\n{detail}")

    def _on_reply(self, message) -> None:
        """Store a reply under its sequence number."""
        sequence, status, reply = decode(bytes(message.data))
        self.replies[sequence] = (status, reply)

    def _call(self, request: tuple) -> tuple:
        """Publish one request and spin until its reply arrives."""
        self.sequence += 1
        message = self.message_type()
        message.data = list(encode((self.sequence, request)))
        self.publisher.publish(message)
        deadline = time.perf_counter() + self.reply_timeout_s
        while self.sequence not in self.replies:
            self.rclpy.spin_once(self.node, timeout_sec=0.001)
            if time.perf_counter() > deadline:
                raise BridgeNodeError(f"node {self.name} did not reply within {self.reply_timeout_s} s")
        return self.replies.pop(self.sequence)

    def request(self, message: tuple) -> tuple[Any, float]:
        """Send one request and wait for its reply; return (reply, round trip in seconds)."""
        began = time.perf_counter()
        status, reply = self._call(message)
        if status != "ok":
            raise BridgeNodeError(f"node {self.name} failed on '{message[0]}':\n{reply}")
        return reply, time.perf_counter() - began

    def publish_clock(self, seconds: float) -> None:
        """Publish the simulation time on /clock."""
        message = self.clock_type()
        message.clock.sec = int(seconds)
        message.clock.nanosec = int(round((seconds - int(seconds)) * 1e9))
        self.clock_publisher.publish(message)

    def close(self) -> None:
        """Tell the node to stop and destroy the rclpy node."""
        try:
            self._call(("close",))
        except Exception:
            pass
        self.node.destroy_node()


def make_transport(kind: str, spec: dict, name: str):
    """Return the transport named ``kind`` (loopback or ros2) connected to a node built from ``spec``."""
    if kind == "loopback":
        return LoopbackTransport(spec, name)
    if kind == "ros2":
        return Ros2Transport(spec, name)
    raise ValueError(f"unknown bridge transport '{kind}' (loopback or ros2)")
