#!/usr/bin/env python3
"""Pure-pursuit drive-out for a single car-frame lookahead point.

Subscribes to a stream of `geometry_msgs/Point32` (one point per camera
frame, expressed in the rear-axle frame: +x forward, +y left).  For each
point we compute the bearing eta = atan2(y, x), pick a lookahead radius
D from the {tangent / default / corner} schedule, and emit

    delta = atan2( 2 W sin(eta), D )

When no fresh point is available we hold the most recent steer for a
short grace window before idling the wheels.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point32
from rcl_interfaces.msg import SetParametersResult
from visualization_msgs.msg import Marker


def _saturate(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class LaneTracer(Node):
    """ROS 2 node that turns a /lookahead_point stream into a drive command.

    Output topics
    -------------
    drive_topic         AckermannDriveStamped
    /lookahead_target   visualization_msgs/Marker (sphere, color = state)

    Marker color schedule (kept stable so visualisers can decode):
        FRESH_POINT → green
        HELD_POINT  → yellow
        BLIND       → red
    """

    STATE_FRESH = "FRESH_POINT"
    STATE_HELD  = "HELD_POINT"
    STATE_BLIND = "BLIND"

    _MARKER_COLORS = {
        STATE_FRESH: (0.0, 1.0, 0.0),
        STATE_HELD:  (1.0, 1.0, 0.0),
        STATE_BLIND: (1.0, 0.0, 0.0),
    }

    def __init__(self) -> None:
        super().__init__("lane_tracer")

        # ── topic wiring ────────────────────────────────────────────────
        self.declare_parameter("lookahead_topic", "/lookahead_point")
        self.declare_parameter("drive_topic",     "/drive")

        # ── geometry & pursuit gains (mirrors the racetrack reference) ──
        self.declare_parameter("wheelbase_m", 0.33)
        self.declare_parameter("pursuit_radius_default_m", 4.0)
        self.declare_parameter("pursuit_radius_corner_m",  1.5)
        self.declare_parameter("pursuit_radius_tangent_m", 3.5)
        self.declare_parameter("eta_corner_deadband_rad",  0.2)
        self.declare_parameter("camera_lateral_offset_m",  0.0)

        # ── safety & cadence ─────────────────────────────────────────────
        self.declare_parameter("cruise_speed_mps",   2.5)
        self.declare_parameter("max_steering_rad",   0.34)
        self.declare_parameter("freshness_window_s", 0.80)
        self.declare_parameter("tick_rate_hz",       33.0)
        self.declare_parameter("idle_when_blind",    True)

        self._refresh_tunables()

        cb_group = MutuallyExclusiveCallbackGroup()
        self._cb_group = cb_group

        look_topic  = str(self.get_parameter("lookahead_topic").value)
        drive_topic = str(self.get_parameter("drive_topic").value)

        self.aim_sub = self.create_subscription(
            Point32, look_topic, self._handle_aim_point, 10,
            callback_group=cb_group,
        )
        self.drive_pub  = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10,
        )
        self.marker_pub = self.create_publisher(
            Marker, "/lookahead_target", 10,
        )

        # ── state ───────────────────────────────────────────────────────
        self._latest_aim:    Optional[tuple] = None  # (x, y) in car frame
        self._latest_aim_ts: Optional[object] = None
        self._held_steer: float = 0.0

        # ── inverse homography exposed for downstream visualisers ───────
        from final_challenge.homography_transformer import build_homography
        self._H = build_homography()
        self.H_inv = np.linalg.inv(self._H)

        # ── periodic control tick ────────────────────────────────────────
        period = 1.0 / max(float(self.tick_rate_hz), 1.0)
        self.control_timer = self.create_timer(
            period, self._tick, callback_group=cb_group,
        )

        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(
            f"LaneTracer up — W={self.wheelbase_m:.2f} m, "
            f"D={self.pursuit_radius_tangent_m:.1f}/{self.pursuit_radius_default_m:.1f}/"
            f"{self.pursuit_radius_corner_m:.1f} m (tan/def/cor), "
            f"v={self.cruise_speed_mps:.1f} m/s"
        )

    # ── parameters ───────────────────────────────────────────────────────
    def _refresh_tunables(self) -> None:
        g = self.get_parameter
        self.wheelbase_m = float(g("wheelbase_m").value)
        self.pursuit_radius_default_m = float(g("pursuit_radius_default_m").value)
        self.pursuit_radius_corner_m  = float(g("pursuit_radius_corner_m").value)
        self.pursuit_radius_tangent_m = float(g("pursuit_radius_tangent_m").value)
        self.eta_corner_deadband_rad  = float(g("eta_corner_deadband_rad").value)
        self.camera_lateral_offset_m  = float(g("camera_lateral_offset_m").value)
        self.cruise_speed_mps         = float(g("cruise_speed_mps").value)
        self.max_steering_rad         = float(g("max_steering_rad").value)
        self.freshness_window_s       = float(g("freshness_window_s").value)
        self.tick_rate_hz             = float(g("tick_rate_hz").value)
        self.idle_when_blind          = bool(g("idle_when_blind").value)

    def _on_param_update(self, params) -> SetParametersResult:
        floats = {
            "wheelbase_m", "pursuit_radius_default_m", "pursuit_radius_corner_m",
            "pursuit_radius_tangent_m", "eta_corner_deadband_rad",
            "camera_lateral_offset_m", "cruise_speed_mps",
            "max_steering_rad", "freshness_window_s", "tick_rate_hz",
        }
        for p in params:
            if p.name in floats:
                setattr(self, p.name, float(p.value))
            elif p.name == "idle_when_blind":
                self.idle_when_blind = bool(p.value)
        return SetParametersResult(successful=True)

    # ── input ────────────────────────────────────────────────────────────
    def _handle_aim_point(self, msg: Point32) -> None:
        x, y = float(msg.x), float(msg.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        # The detector publishes (0, 0) as a sentinel for "no geometry";
        # treat it as if the message hadn't arrived rather than steering
        # toward the rear axle.
        if x == 0.0 and y == 0.0:
            return
        self._latest_aim = (x, y + self.camera_lateral_offset_m)
        self._latest_aim_ts = self.get_clock().now()

    # ── pursuit law ──────────────────────────────────────────────────────
    def _select_lookahead(self, eta_rad: float) -> float:
        if abs(eta_rad) > self.eta_corner_deadband_rad:
            return self.pursuit_radius_corner_m
        return self.pursuit_radius_tangent_m

    def _aim_is_fresh(self) -> bool:
        if self._latest_aim is None or self._latest_aim_ts is None:
            return False
        age_s = (self.get_clock().now() - self._latest_aim_ts).nanoseconds * 1e-9
        return age_s <= self.freshness_window_s

    # ── tick ─────────────────────────────────────────────────────────────
    def _tick(self) -> None:
        fresh = self._aim_is_fresh()
        aim = self._latest_aim if fresh else None

        if aim is None:
            self._publish_marker(self._latest_aim, self.STATE_BLIND)
            if self.idle_when_blind:
                self._send_drive(0.0, 0.0)
            else:
                self._send_drive(self.cruise_speed_mps, self._held_steer)
            return

        ax, ay = aim
        if ax <= 1e-3:
            # Aim-point at/behind the rear axle — pursuit law breaks; coast.
            self._publish_marker(aim, self.STATE_HELD)
            self._send_drive(self.cruise_speed_mps, self._held_steer)
            return

        eta = math.atan2(ay, ax)
        D = self._select_lookahead(eta)
        steer = math.atan2(2.0 * self.wheelbase_m * math.sin(eta), D)
        steer = _saturate(steer, -self.max_steering_rad, self.max_steering_rad)

        self._held_steer = steer
        self._publish_marker(aim, self.STATE_FRESH)
        self._send_drive(self.cruise_speed_mps, steer)

    # ── publishers ───────────────────────────────────────────────────────
    def _send_drive(self, v_mps: float, steer_rad: float) -> None:
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.drive.speed = float(v_mps)
        cmd.drive.steering_angle = float(steer_rad)
        self.drive_pub.publish(cmd)

    def _publish_marker(self, aim: Optional[tuple], state: str) -> None:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "lane_tracer"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        if aim is None:
            m.pose.position.x = 0.0
            m.pose.position.y = 0.0
        else:
            m.pose.position.x = float(aim[0])
            m.pose.position.y = float(aim[1])
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.10
        r, g, b = self._MARKER_COLORS.get(state, (0.5, 0.5, 0.5))
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200_000_000
        self.marker_pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneTracer()
    pool = MultiThreadedExecutor()
    pool.add_node(node)
    try:
        pool.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._send_drive(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
