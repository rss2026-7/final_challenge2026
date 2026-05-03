#!/usr/bin/env python3
"""Synthetic /lookahead_point publisher for exercising LaneTracer offline.

Useful when you want to drive the controller without the full ZED + Hough
pipeline.  The published point is whatever the selected mode dictates,
emitted at a configurable rate.

Usage:
    ros2 run final_challenge test_lane_publisher
    ros2 run final_challenge test_lane_publisher --ros-args -p mode:=swerve

Modes:
    forward     – aim straight ahead (centred on the lane)
    bend_left   – aim ahead-and-left
    bend_right  – aim ahead-and-right
    swerve      – sinusoidal y oscillation around centre
    flicker     – publish for 2 s, then go silent for 1 s (perception drop)
"""
from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point32


class FakeAimEmitter(Node):
    def __init__(self) -> None:
        super().__init__("fake_aim_emitter")

        self.declare_parameter("mode",         "forward")
        self.declare_parameter("aim_x_m",      2.0)
        self.declare_parameter("bend_y_m",     0.4)
        self.declare_parameter("swerve_y_m",   0.3)
        self.declare_parameter("swerve_hz",    0.5)
        self.declare_parameter("publish_rate", 20.0)

        self._mode       = str(self.get_parameter("mode").value)
        self._aim_x      = float(self.get_parameter("aim_x_m").value)
        self._bend_y     = float(self.get_parameter("bend_y_m").value)
        self._swerve_y   = float(self.get_parameter("swerve_y_m").value)
        self._swerve_hz  = float(self.get_parameter("swerve_hz").value)
        rate             = float(self.get_parameter("publish_rate").value)

        self._pub = self.create_publisher(Point32, "/lookahead_point", 10)
        self._timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)
        self._t0 = time.time()

        self.get_logger().info(
            f"FakeAimEmitter up — mode={self._mode}, aim_x={self._aim_x:.2f}m, "
            f"rate={rate:.1f} Hz"
        )

    def _tick(self) -> None:
        elapsed = time.time() - self._t0

        if self._mode == "flicker":
            cycle = elapsed % 3.0
            if cycle > 2.0:
                return
            x, y = self._aim_x, 0.0
        elif self._mode == "forward":
            x, y = self._aim_x, 0.0
        elif self._mode == "bend_left":
            x, y = self._aim_x, +self._bend_y
        elif self._mode == "bend_right":
            x, y = self._aim_x, -self._bend_y
        elif self._mode == "swerve":
            x = self._aim_x
            y = self._swerve_y * math.sin(2.0 * math.pi * self._swerve_hz * elapsed)
        else:
            self.get_logger().warn(
                f"unknown mode '{self._mode}', falling back to 'forward'",
                throttle_duration_sec=2.0,
            )
            x, y = self._aim_x, 0.0

        msg = Point32()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = 0.0
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeAimEmitter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
