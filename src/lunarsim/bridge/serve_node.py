# SPDX-License-Identifier: BSD-3-Clause
"""Node process of the software-in-the-loop bridge: ``python -m lunarsim.bridge.serve_node``.

``--transport loopback --address HOST:PORT --name NAME``: connect to the simulator's bridge binding
(key in the environment variable ``LUNARSIM_BRIDGE_KEY``), receive the spec, build the node
program and serve requests until ``close``.

``--transport ros2 --name NAME`` (untested; needs ROS 2): subscribe to ``/lunarsim/NAME/request``
and answer on ``/lunarsim/NAME/reply``; the first request carries the spec.

Every reply is ``(status, value)``: ``("ready", info)`` after the spec, ``("ok", reply)`` after a
request, ``("error", traceback)`` when the node program raised.
"""

from __future__ import annotations

import argparse
import os
import traceback

from lunarsim.bridge.message_transport import decode, encode


def serve_loopback(address: str, name: str) -> None:
    """Serve one simulator connection until it sends ``close``."""
    from multiprocessing.connection import Client

    from lunarsim.bridge.node_programs import build_node_program, handle_request

    host, port = address.rsplit(":", 1)
    connection = Client((host, int(port)), authkey=bytes.fromhex(os.environ["LUNARSIM_BRIDGE_KEY"]))
    spec = decode(connection.recv_bytes())
    try:
        program = build_node_program(spec)
    except Exception:
        connection.send_bytes(encode(("error", traceback.format_exc())))
        return
    connection.send_bytes(encode(("ready", {"name": name, "pid": os.getpid()})))
    while True:
        try:
            request = decode(connection.recv_bytes())
        except EOFError:
            return
        if request[0] == "close":
            return
        try:
            reply = ("ok", handle_request(program, request))
        except Exception:
            reply = ("error", traceback.format_exc())
        connection.send_bytes(encode(reply))


def serve_ros2(name: str) -> None:
    """Serve requests on ROS 2 topics (untested here)."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import UInt8MultiArray

    from lunarsim.bridge.node_programs import build_node_program, handle_request

    rclpy.init()
    node = Node(f"lunarsim_{name}")
    publisher = node.create_publisher(UInt8MultiArray, f"/lunarsim/{name}/reply", 10)
    state = {"program": None, "running": True}

    def on_request(message) -> None:
        """Answer one request with the same sequence number."""
        sequence, request = decode(bytes(message.data))
        if request[0] == "close":
            status, value = "ok", None
            state["running"] = False
        elif request[0] == "spec":
            try:
                state["program"] = build_node_program(request[1])
                status, value = "ready", {"name": name}
            except Exception:
                status, value = "error", traceback.format_exc()
        else:
            try:
                status, value = "ok", handle_request(state["program"], request)
            except Exception:
                status, value = "error", traceback.format_exc()
        reply = UInt8MultiArray()
        reply.data = list(encode((sequence, status, value)))
        publisher.publish(reply)

    node.create_subscription(UInt8MultiArray, f"/lunarsim/{name}/request", on_request, 10)
    while state["running"] and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


def main() -> None:
    """Parse the arguments and serve."""
    parser = argparse.ArgumentParser(description="LunarSim bridge node (S10).")
    parser.add_argument("--transport", choices=("loopback", "ros2"), required=True)
    parser.add_argument("--address", default="")
    parser.add_argument("--name", required=True)
    arguments = parser.parse_args()
    if arguments.transport == "loopback":
        serve_loopback(arguments.address, arguments.name)
    else:
        serve_ros2(arguments.name)


if __name__ == "__main__":
    main()
