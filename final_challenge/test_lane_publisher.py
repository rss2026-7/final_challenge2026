#!/usr/bin/env python3
"""
Synthetic lane-line publisher for testing BoundaryPurePursuit.

Publishes fake /left_lane_line and /right_lane_line Path messages
so you can exercise the controller without real perception.

Usage:
    ros2 run final_challenge test_lane_publisher

Modes (set via --ros-args -p mode:=<mode>):
    straight    – straight lane lines ahead
    curve_left  – gentle left curve
    curve_right – gentle right curve
    dropout     – publishes for 2 s then stops for 1 s (repeating)
"""

import math
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def make_path_msg(points: List[Tuple[float, float]], stamp) -> Path:
    msg = Path()
    msg.header.stamp = stamp
    msg.header.frame_id = "base_link"
    for x, y in points:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


class TestLanePublisher(Node):
    def __init__(self):
        super().__init__("test_lane_publisher")

        self.declare_parameter("mode", "straight")
        self.declare_parameter("lane_width", 0.6)       # meters between lines
        self.declare_parameter("num_points", 30)
        self.declare_parameter("point_spacing", 0.10)    # meters between points
        self.declare_parameter("curve_radius", 3.0)      # meters (for curve modes)
        self.declare_parameter("publish_rate", 20.0)     # Hz

        self.mode = str(self.get_parameter("mode").value)
        self.lane_width = float(self.get_parameter("lane_width").value)
        self.num_points = int(self.get_parameter("num_points").value)
        self.point_spacing = float(self.get_parameter("point_spacing").value)
        self.curve_radius = float(self.get_parameter("curve_radius").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)

        self.left_pub = self.create_publisher(Path, "/left_lane_line", 10)
        self.right_pub = self.create_publisher(Path, "/right_lane_line", 10)

        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self.publish_tick)
        self.start_time = time.time()

        self.get_logger().info(
            f"TestLanePublisher started  mode={self.mode}  "
            f"lane_width={self.lane_width}  curve_radius={self.curve_radius}"
        )

    def publish_tick(self):
        elapsed = time.time() - self.start_time

        # Dropout mode: publish 2 s on, 1 s off
        if self.mode == "dropout":
            cycle = elapsed % 3.0
            if cycle > 2.0:
                return  # simulate perception dropout

        stamp = self.get_clock().now().to_msg()

        left_pts, right_pts = self._generate_lane(self.mode)

        self.left_pub.publish(make_path_msg(left_pts, stamp))
        self.right_pub.publish(make_path_msg(right_pts, stamp))

    def _generate_lane(self, mode: str):
        half_w = self.lane_width / 2.0

        if mode in ("straight", "dropout"):
            left_pts = []
            right_pts = []
            for i in range(self.num_points):
                x = 0.3 + i * self.point_spacing  # start 30 cm ahead
                left_pts.append((x, half_w))
                right_pts.append((x, -half_w))
            return left_pts, right_pts

        elif mode in ("curve_left", "curve_right"):
            sign = 1.0 if mode == "curve_left" else -1.0
            r = self.curve_radius
            # Center of the turning circle
            cx = 0.0
            cy = sign * r

            left_pts = []
            right_pts = []
            for i in range(self.num_points):
                arc = (0.3 + i * self.point_spacing) / r
                # Centerline point on the arc
                mx = cx + r * math.sin(arc)
                my = cy - sign * r * math.cos(arc)

                # Tangent direction
                tx = math.cos(arc)
                ty = sign * math.sin(arc)

                # Normal (pointing left of tangent)
                nx = -ty
                ny = tx

                left_pts.append((mx + half_w * nx, my + half_w * ny))
                right_pts.append((mx - half_w * nx, my - half_w * ny))
            return left_pts, right_pts

        else:
            self.get_logger().warn(f"Unknown mode '{mode}', defaulting to straight")
            return self._generate_lane("straight")


def main(args=None):
    rclpy.init(args=args)
    node = TestLanePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
